
from typing import Union, Callable, Optional, Generic, TypeVar
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor

Index = Union[int, slice, Tensor]
Number = Union[int, float]
_S = slice(None)


class _FunctionSpace(metaclass=ABCMeta):
    r"""THe base class of function spaces"""
    device: torch.device
    ftype: torch.dtype
    itype: torch.dtype

    # basis
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
    def grad_value(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self) -> Tensor: raise NotImplementedError
    def face_to_dof(self) -> Tensor: raise NotImplementedError

    # interpolation
    def interpolate(self, source: Union[Callable[..., Tensor], Tensor, Number],
                    uh: Tensor, dim: Optional[int]=None, index: Index=_S) -> Tensor:
        raise NotImplementedError

    # function
    def array(self, dim: int=0) -> Tensor:
        GDOF = self.number_of_global_dofs()
        kwargs = {'device': self.device, 'dtype': self.ftype}

        if dim  == 0:
            shape = (GDOF, )
        else:
            shape = (GDOF, dim)

        return torch.zeros(shape, **kwargs)


_FS = TypeVar('_FS', bound=_FunctionSpace)


class Function(Tensor, Generic[_FS]):
    space: _FS

    # NOTE: Named tensors and all their associated APIs are an experimental feature
    # and subject to change. Please do not use them for anything important until
    # they are released as stable.
    @staticmethod
    def __new__(cls, space: _FS, tensor: Tensor) -> Tensor:
        assert isinstance(space, _FunctionSpace)
        tensor = tensor.to(device=space.device, dtype=space.ftype)
        return Tensor._make_subclass(cls, tensor)

    def __init__(self, space: _FS, tensor: Tensor) -> None:
        self.space = space

    def __call__(self, bc: Tensor, index=_S) -> Tensor:
        return self.space.value(self, bc, index)

    # NOTE: Some methods and attributes of Tensor are very similar to those of FunctionSpace.
    # Such as `values()`, `grad`.

    def grad_value(self, bc: Tensor, index=_S):
        return self.space.grad_value(self, bc, index)

    def interpolate_(self, source: Union[Callable[..., Tensor], Tensor, Number],
                     dim: Optional[int]=None, index: Index=_S) -> Tensor:
        return self.space.interpolate(source, self, dim, index)


class FunctionSpace(_FunctionSpace):
    def function(self, tensor: Optional[Tensor]=None, dim: int=0):
        if tensor is None:
            tensor = self.array(dim=dim)

        dof_dim = -1 if dim == 0 else -2
        func_ = Function(self, tensor, dof_dim=dof_dim)

        return func_
