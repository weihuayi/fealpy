
from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Optional

from torch import Tensor

from ..functionspace.space import FunctionSpace as _FunctionSpace

Index = Union[int, slice, Tensor]
_S = slice(None)
CoefLike = Union[float, int, Tensor, Callable[..., Tensor]]


class Integrator(metaclass=ABCMeta):
    r"""@brief The base class for integrators on function spaces."""
    _value: Optional[Tensor]
    _assembly: str

    def __init__(self, index: Index=_S, method='assembly') -> None:
        self.index = index
        self._assembly = method

    def __call__(self, space: _FunctionSpace) -> Tensor:
        if hasattr(self, '_value') and self._value is not None:
            return self._value
        else:
            if not hasattr(self, '_assembly'):
                raise NotImplementedError("Assembly method not defined.")
            if self._assembly == '__call__':
                raise ValueError("Can not use assembly method name '__call__'.")
            self._value = getattr(self, self._assembly)(space)
            return self._value

    @abstractmethod
    def to_global_dof(self, space: _FunctionSpace) -> Tensor:
        """@brief Return the relationship between the integral entities
        and the global dofs."""
        raise NotImplementedError

    def clear(self):
        """@brief Clear the cached value."""
        self._value = None


class CellOperatorIntegrator(Integrator):
    def assembly(self, space: _FunctionSpace) -> Tensor:
        raise NotImplementedError


class FaceOperatorIntegrator(Integrator):
    def assembly(self, space: _FunctionSpace) -> Tensor:
        raise NotImplementedError


class CellSourceIntegrator(Integrator):
    def assembly(self, space: _FunctionSpace) -> Tensor:
        raise NotImplementedError


class FaceSourceIntegrator(Integrator):
    def assembly(self, space: _FunctionSpace) -> Tensor:
        raise NotImplementedError


__all__ = [
    'Integrator',
    'CellOperatorIntegrator',
    'FaceOperatorIntegrator',
    'CellSourceIntegrator',
    'FaceSourceIntegrator'
]
