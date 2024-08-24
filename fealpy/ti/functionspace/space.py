from typing import Union, Callable, Optional, Generic, TypeVar
from abc import ABCMeta, abstractmethod

import taichi as ti 
from .. import numpy as tnp

Field = TypeVar['Field']
Index = Union[int, slice, Field]
Number = Union[int, float]
_dtype = TypeVar['_dtype']
_S = slice(None)


class _FunctionSpace(metaclass=ABCMeta):
    r"""THe base class of function spaces"""
    ftype: _dtype
    itype: _dtype

    # basis
    @abstractmethod
    def basis(self, p: Field, index: Index=_S, **kwargs) -> Field: raise NotImplementedError
    @abstractmethod
    def grad_basis(self, p: Field, index: Index=_S, **kwargs) -> Field: raise NotImplementedError
    @abstractmethod
    def hess_basis(self, p: Field, index: Index=_S, **kwargs) -> Field: raise NotImplementedError

    # values
    @abstractmethod
    def value(self, uh: Field, p: Field, index: Index=_S) -> Field: raise NotImplementedError
    @abstractmethod
    def grad_value(self, uh: Field, p: Field, index: Index=_S) -> Field: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self) -> Field: raise NotImplementedError
    def face_to_dof(self) -> Field: raise NotImplementedError

    # interpolation
    def interpolate(self, source: Union[Callable[..., Field], Field, Number],
                    uh: Field, dim: Optional[int]=None, index: Index=_S) -> Tensor:
        raise NotImplementedError

    # function
    def array(self, dim: int=0) -> Field:
        GDOF = self.number_of_global_dofs()
        kwargs = {'dtype': self.ftype}
        if dim  == 0:
            shape = (GDOF, )
        else:
            shape = (GDOF, dim)

        return tnp.zeros(shape, **kwargs)


_FS = TypeVar('_FS', bound=_FunctionSpace)


class FunctionSpace(_FunctionSpace):
    def function(self, field: Optional[Field]=None, dim: int=0):
        pass
