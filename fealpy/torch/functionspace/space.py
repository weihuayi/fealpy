
from typing import Union, Generic, TypeVar
from abc import ABCMeta, abstractmethod

from torch import Tensor

from ..mesh.mesh_base import Mesh

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, Tensor]
_S = slice(None)


class FunctionSpace(Generic[_MT], metaclass=ABCMeta):

    ### basis
    @abstractmethod
    def basis(self, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError
    @abstractmethod
    def grad_basis(self, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError
    @abstractmethod
    def hess_basis(self, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError

    # values
    @abstractmethod
    def value(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError
    @abstractmethod
    def grad(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError

    # counters
    @abstractmethod
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    @abstractmethod
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    @abstractmethod
    def cell_to_dof(self) -> Tensor: raise NotImplementedError
