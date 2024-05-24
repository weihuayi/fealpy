
from typing import Union, Callable, Optional
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor

Index = Union[int, slice, Tensor]
Number = Union[int, float]
_S = slice(None)


class FunctionSpace(metaclass=ABCMeta):
    r"""THe base class of function spaces"""
    device: torch.device
    ftype: torch.dtype
    itype: torch.dtype

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

    # interpolation
    def interpolate(self, source: Union[Callable[..., Tensor], Tensor, Number],
                    uh: Tensor, dim: Optional[int]=None, index: Index=_S) -> Tensor:
        raise NotImplementedError
