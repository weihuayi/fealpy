
from typing import Optional
from functools import partial

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func, is_scalar, is_tensor, fill_axis
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    NonlinearInt, OpInt, CellInt,
    enable_cache
)


class ScalarNonlinearConvectionIntegrator(NonlinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        if hasattr(coef, 'uh'):
            self.uh = coef.uh
            self.kernel_func = coef.kernel_func
            if not hasattr(coef, 'grad_kernel_func'):
                assert bm.backend_name != "numpy", "In the numpy backend, you must provide a 'grad_kernel_func' method for the coefficient."
                self.grad_kernel_func = None
            else:
                self.grad_kernel_func = coef.grad_kernel_func
        self.q = q
        self.index = index
        self.batched = batched

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
    
    def assembly(self, space: _FS) -> TensorLike:
        uh = self.uh
        coef = self.coef 
        mesh = getattr(space, 'mesh', None)
        bcs, _, _, _, index = self.fetch(space)       
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        uh_ = uh[space.cell_to_dof()]
        A, F = self.auto_grad(space, uh_, coef, batched=self.batched) 

        return A, F

    def cell_integral(self, u, cm, phi, coef, gphi, ws, batched) -> TensorLike:
        val = self.kernel_func(bm.einsum('i, qi -> q', u, phi[0]))

        if is_tensor(coef):
            coef = fill_axis(coef, 3 if batched else 2)
            return bm.einsum(f'q, qid, q, ...qd -> ...i', ws, gphi, val, coef) * cm
        else:
            raise TypeError(f"coef should be Tensor, but got {type(coef)}.")

    def auto_grad(self, space, uh_, coef, batched) -> TensorLike:
        _, ws, phi, gphi, cm, _ = self.fetch(space)
        cell_integral = partial(self.cell_integral, phi=phi, gphi=gphi, ws=ws, batched=batched)
        fn_A = bm.vmap(bm.jacfwd(cell_integral))
        fn_F = bm.vmap(cell_integral)
        return fn_A(uh_, gphi, cm, coef), -fn_F(uh_, gphi, cm, coef)

    