
from typing import Union, Callable, Optional, Generic, TypeVar
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray

Index = Union[int, slice, NDArray]
Number = Union[int, float]
_S = slice(None)


class _FunctionSpace(metaclass=ABCMeta):
    r"""THe base class of function spaces"""
    ftype: np.dtype
    itype: np.dtype

    ### basis
    @abstractmethod
    def basis(self, p: NDArray, index: Index=_S, **kwargs) -> NDArray: raise NotImplementedError
    @abstractmethod
    def grad_basis(self, p: NDArray, index: Index=_S, **kwargs) -> NDArray: raise NotImplementedError
    @abstractmethod
    def hess_basis(self, p: NDArray, index: Index=_S, **kwargs) -> NDArray: raise NotImplementedError

    # values
    @abstractmethod
    def value(self, uh: NDArray, p: NDArray, index: Index=_S) -> NDArray: raise NotImplementedError
    @abstractmethod
    def grad_value(self, uh: NDArray, p: NDArray, index: Index=_S) -> NDArray: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self) -> NDArray: raise NotImplementedError
    def face_to_dof(self) -> NDArray: raise NotImplementedError

    # interpolation
    def interpolate(self, source: Union[Callable[..., NDArray], NDArray, Number],
                    uh: NDArray, dim: Optional[int]=None, index: Index=_S) -> NDArray:
        raise NotImplementedError


_FS = TypeVar('_FS', bound=_FunctionSpace)


class Function(NDArray, Generic[_FS]):
    pass
    

class FunctionSpace(_FunctionSpace):
    def function(self, tensor: Optional[NDArray]=None, dim: int=-1):
        func_ = Function(self, tensor, dim)
        return func_
