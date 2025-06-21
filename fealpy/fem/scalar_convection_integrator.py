from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..utils import is_tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from ..decorator.variantmethod import variantmethod
from .integrator import LinearInt, OpInt, CellInt, enable_cache


class ScalarConvectionIntegrator(LinearInt, OpInt, CellInt):
    r"""The convection integrator for function spaces based on homogeneous meshes."""
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
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarConvectionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index)
        phi = space.basis(bcs, index=index)
        return bcs, ws, phi, gphi, cm, index

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if is_tensor(coef):
            gphi = bm.einsum('cqi...j, cq...j->cqi...' ,gphi, coef)
            result = bilinear_integral(phi, gphi, ws, cm, coef=None, batched=self.batched)
        else:
            raise TypeError(f"coef should be Tensor, but got {type(coef)}.")
        return result
    
    @assembly.register('isopara')
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        rm = space.mesh.reference_cell_measure()
        J = space.mesh.jacobi_matrix(bcs)
        G = space.mesh.first_fundamental_form(J)
        d = bm.sqrt(bm.linalg.det(G))

        if is_tensor(coef):
            gphi = bm.einsum('cqi...j, cq...j->cqi...' ,gphi, coef)
            result = bm.einsum('q, cqi, cqj , cq -> cij', ws*rm, phi, gphi, d)
        else:
            raise TypeError(f"coef should be Tensor, but got {type(coef)}.")
        return result
