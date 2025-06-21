from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from ..decorator.variantmethod import variantmethod
from .integrator import LinearInt, OpInt, CellInt, enable_cache


class ScalarMassIntegrator(LinearInt, OpInt, CellInt):
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
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return bcs, ws, phi, cm, index

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(phi, phi, ws, cm, val, batched=self.batched)

    @assembly.register('semilinear')
    def assembly(self, space: _FS) -> TensorLike:
        uh = self.uh
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val_A = coef.grad_func(uh(bcs))  #(C, Q)
        val_F = -coef.func(uh(bcs))      #(C, Q)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        coef_A = get_semilinear_coef(val_A, coef)
        coef_F = get_semilinear_coef(val_F, coef)

        return bilinear_integral(phi, phi, ws, cm, coef_A, batched=self.batched), \
               linear_integral(phi, ws, cm, coef_F, batched=self.batched)

    @assembly.register('isopara')
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        """
        等参有限元质量矩阵组装
        """
        index = self.entity_selection(indices)
        mesh = getattr(space, 'mesh', None)

        rm = mesh.reference_cell_measure()
        cm = mesh.entity_measure('cell', index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = mesh.jacobi_matrix(bcs, index=index)
        G = mesh.first_fundamental_form(J) 
        d = bm.sqrt(bm.linalg.det(G))
        phi = space.basis(bcs)
        M = bm.einsum('q, cqi, cqj, cq -> cij', ws*rm, phi, phi, d) #(NC, ldof, ldof)
        return M
