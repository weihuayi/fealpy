
from typing import Optional
from functools import partial

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func, is_scalar, is_tensor, fill_axis
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    SemilinearInt, OpInt, CellInt,
    enable_cache,
    CoefLike
)

class NonlinearElasticIntegrator(SemilinearInt, OpInt, CellInt):
    r"""The nonlinear elastic integrator for function spaces based on homogeneous meshes."""
    def __init__(self, material, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.material = material

        if hasattr(material, 'uh'):
            self.uh = material.uh
            self.stress_func = material.stress_func
            if not hasattr(material, 'grad_stress_func'):
                assert bm.backend_name != "numpy", "In the numpy backend, you must provide a 'grad_kernel_func' method for the coefficient."
                self.grad_stress_func = None
            else:
                self.grad_stress_func = material.elastic_matrix
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
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space)       # gphi.shape ==[NC, NQ, ldof, dof_numel, GD]
        B = self.material.strain_matrix(dof_priority=space.dof_priority, gphi=gphi)
        self._B = B
        self._bcs = bcs
        if self.grad_stress_func is not None:
            val_A = self.grad_stress_func(bcs)         # [NC, NQ]          
            A = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, val_A, B)
            val_F = -uh.grad_value(bcs)                    # [NC, NQ, dof_numel, GD]
            F = linear_integral(gphi, ws, cm, val_F, batched=self.batched)
        else:
            uh_ = uh[space.cell_to_dof()]
            A, F = self.auto_grad(space, uh_, bcs=bcs, batched=self.batched) 

        return A, F
    
    def cell_integral(self, u, B, cm, ws, batched) -> TensorLike:
        bcs = self._bcs
        val = self.stress_func(bm.einsum('cqkl, cl... -> cqk', B, u), bcs)
        return bm.einsum('q, c, cqki, cqk -> ci', ws, cm, B, val)

    def auto_grad(self, space, uh_, bcs, batched) -> TensorLike:
        _, ws, gphi, cm, _ = self.fetch(space)
        B = self._B
        fn_A = bm.vmap(bm.jacfwd(                         
            partial(self.cell_integral, ws=ws, batched=batched)
            ))
        fn_F = bm.vmap(
            partial(self.cell_integral, ws=ws, batched=batched)
        )
        return fn_A(uh_, B, cm), -fn_F(uh_, B, cm)
        