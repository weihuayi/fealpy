
from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Optional, Any

from torch import Tensor

from ..functionspace.space import FunctionSpace as _FS

Index = Union[int, slice, Tensor]
_S = slice(None)
CoefLike = Union[float, int, Tensor, Callable[..., Tensor]]


def enable_cache(meth: Callable[[Any, _FS], Tensor]) -> Callable[[Any, _FS], Tensor]:
    def wrapper(self, space: _FS) -> Tensor:
        if getattr(self, '_cache', None) is None:
            self._cache = {}
        _cache = self._cache
        key = meth.__name__ + '_' + str(id(space))
        if key not in _cache:
            _cache[key] = meth(self, space)
        return _cache[key]
    return wrapper


class Integrator(metaclass=ABCMeta):
    r"""@brief The base class for integrators on function spaces."""
    _value: Optional[Tensor]
    _assembly: str

    def __init__(self, method='assembly') -> None:
        self._assembly = method

    def __call__(self, space: _FS) -> Tensor:
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
    def to_global_dof(self, space: _FS) -> Tensor:
        """Return the relationship between the integral entities
        and the global dofs."""
        raise NotImplementedError

    def clear(self, space_level=True) -> None:
        """Clear the cache of the integrators.

        Args:
            space_level (bool, optional): Whether to clear cache for space,
            such as basis, entity measure and bc points. Defaults to False.
            If `False`, only clear the cache of the integral result.
        """
        self._value = None

        if space_level:
            if hasattr(self, '_cache'):
                self._cache.clear()


class CellOperatorIntegrator(Integrator):
    def assembly(self, space: _FS) -> Tensor:
        raise NotImplementedError


class FaceOperatorIntegrator(Integrator):
    def assembly(self, space: _FS) -> Tensor:
        raise NotImplementedError


class CellSourceIntegrator(Integrator):
    def assembly(self, space: _FS) -> Tensor:
        raise NotImplementedError


class FaceSourceIntegrator(Integrator):
    def assembly(self, space: _FS) -> Tensor:
        raise NotImplementedError


__all__ = [
    'Integrator',
    'CellOperatorIntegrator',
    'FaceOperatorIntegrator',
    'CellSourceIntegrator',
    'FaceSourceIntegrator'
]
