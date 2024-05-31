from typing import Optional

import numpy as np
import jax 
import jax.numpy as jnp

from .integrator import CellOperatorIntegrator, _S, Index, CoefLike, Tensor


class ScalarDiffusionIntegrator(CellOperatorIntegrator):
    r"""The diffusion integrator for function spaces on homogeneous meshes."""
    def __init__(self, c: Optional[CoefLike]=None, q: int=3, *,
            index: Index=_S,
            batched: bool=False,
            method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(index=index, method=method)
        self.coef = c
        self.q = q
        self.batched = batched

    def to_global_dof(self, sapce) -> Tensor:
        return space.cell_to_dof()[self.index]

    def fast_assembly(self, space) -> Tensor:
        assert self.coef == None
        mesh = space.mesh
        cm = mesh.entity_measure()

        qf = mesh.integrator(self.q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        R = space.grad_basis(bcs, varialbes='u') # (NQ, ldof, TD+1)

        M = jnp.enisum('q, qik, qjl->ijkl', ws, R, R)

        glambda = mesh.grad_lambda()

        A = jnp.enisum('ijkl, ckm, clm->cij', M, glambda, glambda, cm)
        return A


