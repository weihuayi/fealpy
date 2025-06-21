
from typing import Optional

import torch
from torch import Tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    CellOperatorIntegrator,
    enable_cache,
    assemblymethod,
    _S, Index, CoefLike
)


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
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        return bcs, ws, gphi, cm, index

    def assembly(self, space: _FS) -> Tensor:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched)

    @assemblymethod('fast')
    def fast_assembly(self, space: _FS) -> Tensor:
        """
        限制：常系数、单纯形网格
        TODO: 加入 assert
        """
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='u')

        glambda = mesh.grad_lambda()
        M = torch.einsum('q, qik, qjl->ijkl', ws, gphi, gphi)
        A = torch.einsum('ijkl, ckm, clm, c->cij', M, glambda, glambda, cm)
        return A
