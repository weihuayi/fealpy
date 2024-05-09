
from typing import Union
from abc import ABCMeta, abstractmethod

from torch import Tensor

from ..mesh import Mesh

Index = Union[int, slice, Tensor]
_S = slice(None)


class FunctionSpace(metaclass=ABCMeta):
    r"""THe base class of function spaces"""
    mesh: Mesh

    ### basis
    @abstractmethod
    def basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor: raise NotImplementedError
    @abstractmethod
    def grad_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor: raise NotImplementedError
    @abstractmethod
    def hess_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor: raise NotImplementedError

    # values
    @abstractmethod
    def value(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError
    @abstractmethod
    def grad(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self) -> Tensor: raise NotImplementedError
