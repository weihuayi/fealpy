
from typing import Union, Callable, Optional, Any, TypeVar, Tuple, Dict

from ..typing import TensorLike, CoefLike
from ..functionspace.space import FunctionSpace as _FS

__all__ = [
    'Integrator',
    'NonlinearInt',
    'SemilinearInt',
    'LinearInt',
    'OpInt',
    'SrcInt',
    'CellInt',
    'FaceInt',
]

_Meth = TypeVar('_Meth', bound=Callable[..., Any])


class IntegratorMeta(type):
    def __init__(self, name: str, bases: Tuple[type, ...], dict: Dict[str, Any], /, **kwds: Any):
        for meth_name, meth in dict.items():
            if callable(meth):
                if hasattr(meth, '__call_name__'):
                    call_name = getattr(meth, '__call_name__')
                    if call_name is None:
                        call_name = meth_name
                    if not hasattr(self, '_assembly_map'):
                        self._assembly_map = {}
                    self._assembly_map[call_name] = meth_name

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
    def decorator(meth: _Meth) -> _Meth:
        meth.__call_name__ = call_name
        return meth
    return decorator


def enable_cache(meth: _Meth) -> _Meth:
    """A decorator indicating that the method should be cached by its `space` arg.

    This is useful for assembly methods supporting coefficient and source to
    fetch the data like the basis of space and the measurement of mesh entities.
    Redundant computation can be avoided after coef or source are changed.

    Note that as the result of the integral is automatically cached, assembly
    methods producing determinant values for a space is unnecessary to use cache.
    """
    def wrapper(self, space: _FS) -> TensorLike:
        if getattr(self, '_cache', None) is None:
            self._cache = {}
        _cache = self._cache
        key = meth.__name__ + '_' + str(id(space))
        if key not in _cache:
            _cache[key] = meth(self, space)
        return _cache[key]
    return wrapper


class Integrator(metaclass=IntegratorMeta):
    """The base class for integrators on function spaces."""
    _value: Optional[TensorLike] = None
    _assembly_map: Dict[str, str] = {}

    def __init__(self, method='assembly') -> None:
        if method not in self._assembly_map:
            raise ValueError(f"No assembly method is registered as '{method}'.")
        self._assembly = self._assembly_map[method]

    def __call__(self, space: _FS) -> TensorLike:
        if hasattr(self, '_value') and self._value is not None:
            return self._value
        else:
            if not hasattr(self, '_assembly'):
                raise NotImplementedError("Assembly method not defined.")
            if self._assembly == '__call__':
                raise ValueError("Can not use assembly method name '__call__'.")
            self._value = getattr(self, self._assembly)(space)
            return self._value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._assembly})"

    def to_global_dof(self, space: _FS) -> TensorLike:
        """Return the relationship between the integral entities
        and the global dofs."""
        raise NotImplementedError

    @assemblymethod('assembly')
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
            if hasattr(self, '_cache'):
                self._cache.clear()


# These Integrator classes are for type checking

class NonlinearInt(Integrator):
    """### Nonlinear Integrator
    Base class for integrators without linearity requirement."""
    pass


class SemilinearInt(Integrator):
    """### Semilinear Integrator
    Base class for integrators generating integration linear to test functions `v`."""
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
