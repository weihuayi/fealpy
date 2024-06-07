
from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Optional, Any, TypeVar

from torch import Tensor

from ..functionspace.space import FunctionSpace as _FS

Index = Union[int, slice, Tensor]
_S = slice(None)
CoefLike = Union[float, int, Tensor, Callable[..., Tensor]]
_Meth = TypeVar('_Meth', bound=Callable[..., Any])


def enable_cache(meth: _Meth) -> _Meth:
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

    def clear(self, result_only=True) -> None:
        """Clear the cache of the integrators.

        Args:
            result_only (bool, optional): Whether to only clear the cached result.
            Other cache may be basis, entity measures, bc points... Defaults to True.
            If `False`, anything cached will be cleared.
        """
        self._value = None

        if not result_only:
            if hasattr(self, '_cache'):
                self._cache.clear()


class OperatorIntegrator(Integrator):
    coef: Optional[CoefLike]

    def assembly(self, space: _FS) -> Tensor:
        raise NotImplementedError

    def set_coef(self, coef: Optional[CoefLike]=None, /) -> None:
        """Set a new coefficient of the equation to the integrator.
        This will clear the cached result.

        Args:
            coef (CoefLike | None, optional): Tensor function or Tensor. Defaults to None.
        """
        self.coef = coef
        self.clear()


class SourceIntegrator(Integrator):
    source: Optional[CoefLike]

    def assembly(self, space: _FS) -> Tensor:
        raise NotImplementedError

    def set_source(self, source: Optional[CoefLike]=None, /) -> None:
        """Set a new source term of the equation to the integrator.
        This will clear the cached result.

        Args:
            source (CoefLike | None, optional): Tensor function or Tensor. Defaults to None.
        """
        self.source = source
        self.clear()


# These Integrator classes are for type checking

class CellOperatorIntegrator(OperatorIntegrator):
    pass

class CellSourceIntegrator(SourceIntegrator):
    pass

class FaceOperatorIntegrator(OperatorIntegrator):
    pass

class FaceSourceIntegrator(SourceIntegrator):
    pass


__all__ = [
    'Integrator',
    'OperatorIntegrator',
    'SourceIntegrator',

    'CellOperatorIntegrator',
    'FaceOperatorIntegrator',
    'CellSourceIntegrator',
    'FaceSourceIntegrator'
]
