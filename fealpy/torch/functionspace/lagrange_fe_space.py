
from typing import Union, Generic, TypeVar

import torch
from torch import Tensor

from ..mesh.mesh_base import MeshBase
from .dofs import LinearMeshCFEDof


_dtype = torch.dtype
_MT = TypeVar('_MT', bound=MeshBase)
Index = Union[int, slice, Tensor]
_S = slice(None)


class LagrangeFESpace():
    def __new__(self, mesh: _MT, p: int=1, ctype: str='C'):
        if ctype == 'C':
            return _CFEDof_LagrangeFESpace(mesh, p)
        else:
            raise NotImplementedError


class _CFEDof_LagrangeFESpace(LinearMeshCFEDof):
    r"""Lagrange Finite Element Space"""
    def __init__(self, mesh: _MT, p: int=1) -> None:
        super().__init__(mesh, p)

    @property
    def ftype(self) -> _dtype: return self.mesh.ftype
    @property
    def itype(self) -> _dtype: return self.mesh.itype
    @property
    def TD(self) -> int: return self.mesh.top_dimension()
    @property
    def GD(self) -> int: return self.mesh.geo_dimension()

    def basis(self, bc: Tensor, index: Index=_S):
        return self.mesh.shape_function(bc, index)

    def grad_basis(self, bc: Tensor, index: Index=_S):
        return self.mesh.grad_shape_function(bc, index)

    def hess_basis(self, bc: Tensor, index: Index=_S):
        return self.mesh.hess_shape_function(bc, index)

    def value(self, uh: Tensor, bc: Tensor, index: Index=_S) -> Tensor:
        ...
