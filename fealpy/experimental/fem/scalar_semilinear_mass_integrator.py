from typing import Optional
from functools import partial

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    SemilinearInt, OpInt, CellInt,
    enable_cache,
    CoefLike
)


class ScalarSemilinearMassIntegrator(SemilinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
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
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return bcs, ws, phi, cm, index
    
    def assembly(self, space: _FS) -> TensorLike:
        uh = self.uh
        coef = self.coef 
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        
        if self.grad_kernel_func is not None:
            val_A = self.grad_kernel_func(uh(bcs))
            coef_A = get_semilinear_coef(val_A, coef)
            A = bilinear_integral(phi, phi, ws, cm, coef_A, batched=self.batched)
            val_F = -self.kernel_func(uh(bcs)) 
            coef_F = get_semilinear_coef(val_F, coef)
            F = linear_integral(phi, ws, cm, coef_F, batched=self.batched)
        else:
            uh_ = self.uh[space.cell_to_dof()]
            val = self.kernel_func(uh_)
            coef = get_semilinear_coef(val, coef)
            A, F = self.auto_grad(space, coef) 

        return A, F
    
    def cell_integral_A(self, ws, phi, u) -> TensorLike:

        return bm.einsum(f'q, qi, qj, ...j -> ...i', ws, phi[0], phi[0], u)
    
    def cell_integral_F(self, ws, phi, u) -> TensorLike:

        return bm.einsum(f'q, qi, ...i -> ...i', ws, phi[0], u)
    
    def auto_grad(self, space, val) -> TensorLike:

        _, ws, phi, cm, _ = self.fetch(space)
        fn_A = bm.vmap(bm.jacfwd(                         
            partial(self.cell_integral_A, ws, phi)
            ))
        fn_F = bm.vmap(
            partial(self.cell_integral_F, ws, phi)
        )
        return bm.einsum('...cij, c -> ...cij', fn_A(val), cm),\
               -bm.einsum('...ci, c -> ...ci', fn_F(val), cm)
