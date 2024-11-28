
from typing import Callable, Optional, Any, TypeVar, Tuple, Dict

from .. import logger
from ..typing import TensorLike, CoefLike
from ..functionspace.space import FunctionSpace as _FS
from ..utils import ftype_memory_size

__all__ = [
    'Integrator',
    'NonlinearInt',
    'LinearInt',
    'OpInt',
    'SrcInt',
    'CellInt',
    'FaceInt',
]

Self = TypeVar('Self')


class IntegratorMeta(type):
    def __init__(self, name: str, bases: Tuple[type, ...], dict: Dict[str, Any], /, **kwds: Any):
        if not hasattr(self, '_assembly_name_map'):
            self._assembly_name_map = {}

        for meth_name, meth in dict.items():
            if callable(meth) and hasattr(meth, '__call_name__'):
                call_name = getattr(meth, '__call_name__')
                if call_name is None:
                    call_name = meth_name
                self._assembly_name_map[call_name] = meth_name

        return type.__init__(self, name, bases, dict, **kwds)


def assemblymethod(call_name: Optional[str]=None):
    """A decorator registering the method as an assembly method.

    Requires that the metaclass is IntegratorMeta or derived from it.

    Parameters:
        call_name (str, optional): The name for the users to choose the assembly method.
            Use the name of the method if None. Defaults to None.

    Example:
    ```
        class MyIntegrator(Integrator):
            def assembly(self, space: _FS) -> Tensor:
                # 'assembly' is the default assembly method,
                # naturally registered to name 'assembly'.
                return integral

            @assemblymethod('my')
            def my_assembly(self, space: _FS) -> Tensor:
                # code for getting local integral tensor
                return integral
    ```
    Then in the initialization of an integrator instance, use
    ```
        integrator = MyIntegrator(method='my')
    ```
    to specify the 'my_assembly' as the assembly method called in the __call__.
    """
    if call_name in {'__call__',}:
        raise ValueError(f"Can not use assembly method name '{call_name}'.")
    def decorator(meth: Self) -> Self:
        meth.__call_name__ = call_name
        return meth
    return decorator


def enable_cache(func: Self) -> Self:
    """A decorator indicating that the method should be cached by its `space` arg.

    This is useful for assembly methods supporting coefficient and source to
    fetch the data like the basis of space and the measurement of mesh entities.
    Redundant computation can be avoided after coef or source are changed.

    Use `Integrator.keep_data(True)` to enable the cache.
    """
    def wrapper(integrator_obj, space: _FS) -> TensorLike:
        if not integrator_obj._keep_data:
            return func(integrator_obj, space)

        assert hasattr(integrator_obj, '_cache')
        _cache = integrator_obj._cache
        key = (func.__name__, id(space))

        if key in _cache:
            return _cache[key]

        data = func(integrator_obj, space)
        _cache[key] = data

        return data

    return wrapper


class Integrator(metaclass=IntegratorMeta):
    """The base class for integrators on function spaces."""
    _assembly_name_map: Dict[str, str] = {}

    def __init__(self, method='assembly', keep_data=False, keep_result=False, max_result_MB: float = 256) -> None:
        if method not in self._assembly_name_map:
            raise ValueError(f"No assembly method is registered as '{method}'. "
                             f"For {self.__class__.__name__}, only the following options are available: "
                             f"{', '.join(self._assembly_name_map.keys())}.")

        self._method = method
        self._cache: Dict[Tuple[str, int], Any] = {}
        self._max_result_MB = max_result_MB
        self.keep_data(keep_data)
        self._value: Optional[TensorLike] = None
        self.keep_result(keep_result, None)

    def keep_result(self, status_on=True, /, max_result_MB: Optional[float] = None):
        self._keep_result = status_on
        if status_on:
            if max_result_MB is not None:
                if max_result_MB <= 0:
                    raise ValueError("max_result_MB should be positive.")
                self._max_result_MB = max_result_MB

            if self._result_size_mb() > self._max_result_MB:
                logger.warning(f"{self.__class__.__name__}: Result is larger ({self._result_size_mb():.2f} Mb) "
                               f"than the new max_result_MB ({self._max_result_MB:.2f} Mb), "
                               "and will be cleared automatically.")
                self._value = None
        else:
            self._value = None
        return self

    def keep_data(self, status_on=True, /):
        self._keep_data = status_on
        if not status_on:
            self._cache.clear()
        return self

    def _result_size_mb(self):
        if self._value is None:
            return 0.
        else:
            return ftype_memory_size(self._value, unit='mb')

    def run(self, space: _FS) -> TensorLike:
        if self._value is not None:
            return self._value
        else:
            meth = getattr(self, self._assembly_name_map[self._method], None)
            value = meth(space)

            if self._keep_result:
                memory_size = ftype_memory_size(value, unit='mb')
                if memory_size > self._max_result_MB:
                    logger.info(f"{self.__class__.__name__}: Result is not kept as it is larger ({memory_size:.2f} Mb) "
                                f"than the max_result_MB ({self._max_result_MB:.2f} Mb).")
                else:
                    self._value = value
                    logger.debug(f"{self.__class__.__name__}: Result size: {memory_size:.2f} Mb.")

            return value

    __call__ = run

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._method})"

    def to_global_dof(self, space: _FS) -> TensorLike:
        """Return the relationship between the integral entities
        and the global dofs."""
        raise NotImplementedError

    @assemblymethod('assembly') # The default assemply method
    def assembly(self, space: _FS) -> TensorLike:
        raise NotImplementedError

    def clear(self, result_only=True) -> None:
        """Clear the cache of the integrators.

        Parameters:
            result_only (bool, optional): Whether to only clear the cached result.
                Other cache may be basis, entity measures, bc points... Defaults to True.
                If `False`, anything cached will be cleared.
        """
        self._value = None

        if not result_only:
            self._cache.clear()


# These Integrator classes are for type checking

class NonlinearInt(Integrator):
    """### Nonlinear Integrator
    Base class for integrators without linearity requirement."""
    pass


class LinearInt(Integrator):
    """### Linear Integrator
    Base class for integrators generating integration linear to both `u` and `v`."""
    pass


class OpInt(Integrator):
    """### Operator Integrator
    Base class for integrators involving both the trail function `u` and test function `v`."""
    pass


class SrcInt(Integrator):
    """### Source Integrator
    Base class for integrators involving the test function `v` only."""
    pass


class CellInt(Integrator):
    """### Cell Integrator
    Base class for integrators that integrate over mesh cells."""
    pass


class FaceInt(Integrator):
    """### Face Integrator
    Base class for integrators that integrate over mesh faces."""
    pass

class EdgeInt(Integrator):
    """### Edge Integrator
    Base class for integrators that integrate over mesh edges."""
    pass
