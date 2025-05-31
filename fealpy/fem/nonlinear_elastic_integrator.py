
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

class NonlinearElasticIntegrator(NonlinearInt, OpInt, CellInt):
    r"""The nonlinear elastic integrator for function spaces based on homogeneous meshes."""
    def __init__(self, coef, material, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.material = material
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
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        gphi = space.grad_basis(bcs, index=index, variable='x')
        return bcs, ws, gphi, cm, index
    

    def assembly(self, space: _FS) -> TensorLike:
        uh = self.uh
        scalar_space = space.scalar_space
        mesh = getattr(space, 'mesh', None)
        coef = self.coef 
        

        bcs, ws, gphi, cm, index = self.fetch(scalar_space)       # gphi.shape ==[NC, NQ, ldof, dof_numel]
        B = self.material.strain_matrix(dof_priority=space.dof_priority, gphi=gphi)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if self.grad_kernel_func is not None:
            bcs, ws, gphi, cm, index = self.fetch(space)
            val_A = self.grad_kernel_func(bcs)         # [NC, NQ] 
            coef_A = get_semilinear_coef(val_A, coef)        
            A = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, coef_A, B)
            val_F = -uh.grad_value(bcs)                    # [NC, NQ, dof_numel, GD]
            coef_F = get_semilinear_coef(val_F, coef)
            F = linear_integral(gphi, ws, cm, coef_F, batched=self.batched)
        else:
            uh_ = uh[space.cell_to_dof()]
            A, F = self.auto_grad(space, uh_, B=B, coef=coef, batched=self.batched) 

        return A, F
    
    def cell_integral(self, u, gphi, B, cm, coef, ws, batched) -> TensorLike:
        val = self.kernel_func(guh=bm.einsum('qikl, i -> qkl', gphi, u))
        if coef is None:
            return bm.einsum('q, qki, qk -> i', ws, B, val) * cm
        
        if is_scalar(coef):
            return bm.einsum('q, qki, qk -> i', ws, B, val) * cm * coef
        
        if is_tensor(coef):
            coef = fill_axis(coef, 3 if batched else 2)
            return bm.einsum(f'q, qki, qk, ...qi -> ...i', ws, B, val, coef) * cm

    def auto_grad(self, space, uh_, B, coef, batched) -> TensorLike:
        _, ws, gphi, cm, _ = self.fetch(space)

        if is_scalar(coef) or coef is None:
            cell_integral = partial(self.cell_integral, coef=coef, ws=ws, batched=batched) 
        else:
            cell_integral = partial(self.cell_integral, ws=ws, batched=batched)

        fn_A = bm.vmap(bm.jacfwd(cell_integral))
        fn_F = bm.vmap(cell_integral)
        if is_scalar(coef) or coef is None:
            return fn_A(uh_, gphi, B, cm), -fn_F(uh_, gphi, B, cm)
        else:
            return fn_A(uh_, gphi, B, cm, coef), -fn_F(uh_, gphi, B, cm, coef)

    
    
    