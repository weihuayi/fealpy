from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from ..decorator.variantmethod import variantmethod
from .integrator import LinearInt, OpInt, CellInt, enable_cache


class CouplingMassIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return (space[0].cell_to_dof()[self.index],
                space[1].cell_to_dof()[self.index])

    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]
        q = self.q
        index = self.index

        mesh = getattr(space0, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space0.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi0 = space0.basis(bcs, index=index)
        phi1 = space1.basis(bcs, index=index)
        return bcs, ws, phi0, phi1, cm, index

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space " 
        coef = self.coef
        mesh = getattr(space[0], 'mesh', None)

        bcs, ws, phi0, phi1, cm, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(phi1, phi0, ws, cm, val, batched=self.batched)

    @assembly.register('isopara')
    def assembly(self, space: _FS) -> TensorLike:
        """
        曲面等参有限元积分子组装
        """
        space0 = space[0]
        space1 = space[1]
        
        coef = self.coef
        index = self.index
        mesh = getattr(space[0], 'mesh', None)
        rm = space1.mesh.reference_cell_measure()

        q = space[0].p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        J = space1.mesh.jacobi_matrix(bcs, index=index)
        G = space1.mesh.first_fundamental_form(J) 
        d = bm.sqrt(bm.linalg.det(G))

        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        phi0 = space0.basis(bcs, index=index)
        phi1 = space1.grad_basis(bcs ,index=index)
        phi0 = phi0.reshape(*phi0.shape[:3], -1) # (C, Q, I, dof_numel)
        phi1 = phi1.reshape(*phi1.shape[:3], -1) # (C, Q, J, dof_numel)

        if coef is None:
            result = bm.einsum('q, cqim, cqjm, cq -> cij', ws*rm, phi1, phi0, d)
        elif isinstance(coef, (int, float)):
            result = bm.einsum('q, cqim, cqjm, cq -> cij', ws*rm, phi1, phi0, d) * val
        else:
            raise NotImplementedError("The 'isopara' method only support constant coef.")
        return result

    
