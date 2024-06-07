
from typing import Optional

from torch import Tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import CellOperatorIntegrator, _S, Index, CoefLike, enable_cache


class ScalarDiffusionIntegrator(CellOperatorIntegrator):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, coef: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> Tensor:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS, variable: str='x') -> Tensor:
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable=variable)
        return bcs, ws, gphi, cm, index

    def assembly(self, space: _FS) -> Tensor:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched)

    def fast_assembly(self, space: _FS) -> Tensor:
        """
        限制：常系数、单纯形网格
        TODO: 加入 assert
        """
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space, variable='u')
        M = jnp.enisum('q, qik, qjl->ijkl', ws, gphi, gphi)
        glambda = mesh.grad_lambda()
        A = jnp.enisum('ijkl, ckm, clm->cij', M, glambda, glambda, cm)
        return A
