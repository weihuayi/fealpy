
from typing import Union, Optional, Any, TypeVar, Tuple, List, Dict, Iterable
from copy import deepcopy

from .. import logger
from ..typing import TensorLike, Index, CoefLike, _S
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
    'ConstIntegrator',
    'GroupIntegrator'
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
    """The base class for integrators.

    All integrators should inplement the `assembly` and `to_global_dof` method.
    These two methods indicate two functions of an integrator: calculation of the integral,
    and getting relationship between local DoFs and global DoFs, repectively.
    Users can customize an integrator by implementing them in a subclass.

    ### Multiple Assembly Methods

    Users can develop multiple assembly implementation for a single integrator
    by applying the `assemblymethod` decorator.
    They can be chosen by the `method` parameter when initializing an integrator object.
    See `integrator.assemblymethod` for details.

    ### Using Cache

    The `enable_cache` decorator is a simple tool provided to cache some integral materials.
    This may be useful for unchanged integrators in an iteration algorithm.
    See `integrator.enable_cache` for details.
    """
    _assembly_name_map: Dict[str, str] = {}

    def __init__(self, method='assembly', keep_data=False, keep_result=False, max_result_MB: float = 256) -> None:
        if method not in self._assembly_name_map:
            raise ValueError(f"No assembly method is registered as '{method}'. "
                             f"For {self.__class__.__name__}, only the following options are available: "
                             f"{', '.join(self._assembly_name_map.keys())}.")

        self._method = method
        self._cache: Dict[Tuple[str, int], Any] = {}
        self.keep_data(keep_data)
        self._max_result_MB = max_result_MB
        self._cached_output: Optional[TensorLike] = None
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
                self._cached_output = None
        else:
            self._cached_output = None
        return self

    def keep_data(self, status_on=True, /):
        self._keep_data = status_on
        if not status_on:
            self._cache.clear()
        return self

    def _result_size_mb(self):
        if self._cached_output is None:
            return 0.
        else:
            return ftype_memory_size(self._cached_output, unit='mb')

    def set_index(self, index: Optional[Index], /) -> None:
        """Set the index of integral entity. Skip if `None`."""
        if index is not None:
            self.index = index
            self.clear(result_only=False)

    def __call__(self, space: _FS) -> TensorLike:
        if self._cached_output is not None:
            return self._cached_output
        else:
            meth = getattr(self, self._assembly_name_map[self._method], None)
            value = meth(space)

            if self._keep_result:
                memory_size = ftype_memory_size(value, unit='mb')
                if memory_size > self._max_result_MB:
                    logger.info(f"{self.__class__.__name__}: Result is not kept as it is larger ({memory_size:.2f} Mb) "
                                f"than the max_result_MB ({self._max_result_MB:.2f} Mb).")
                else:
                    self._cached_output = value
                    logger.debug(f"{self.__class__.__name__}: Result size: {memory_size:.2f} Mb.")

            return value

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
        self._cached_output = None

        if not result_only:
            self._cache.clear()

    ### Operations

    def __add__(self, other: 'Integrator'):
        if isinstance(other, Integrator):
            return GroupIntegrator(self, other)
        else:
            return NotImplemented

    __iadd__ = __add__


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


##################################################
### Integral Utils
##################################################

class ConstIntegrator(Integrator):
    """An integral item with given values.

    ConstIntegrator wrap a given TensorLike object as an Integrator type.
    The `to_gdof` is optional but must be provided if `to_global_dof` is needed.
    Index of entity is ignored if `enable_index` is False.
    """
    def __init__(self, value: TensorLike, to_gdof: Optional[TensorLike] = None, *,
                 index: Index = _S, enable_index: bool = False):
        super().__init__('assembly', False, False)
        self.value = value
        self.to_gdof = to_gdof
        self.set_index(index)
        self._enable_index = enable_index

    def to_global_dof(self, space: _FS):
        if self.to_gdof is None:
            raise RuntimeError("to_gdof not defined for ConstIntegrator.")
        return self.to_gdof

    def assembly(self, space: _FS):
        if self._enable_index:
            return self.value[self.index]
        else:
            return self.value


class GroupIntegrator(Integrator):
    """Combine multiple integral items as one.

    GroupIntegrator requires all sub-integrators have the same to_global_dof output.
    That is to say, all integral items must have the same domains and local-global relationship for DoFs.

    Managed by the GroupIntegrator, as a result, sub-integrators ignored their
    own to_global_dof, outputing integrals only.
    The grouped integrator takes the first integrator's `to_global_dof` as its implementation.

    Note: All sub-integrators' index will be replaced by the `index` parameter.
    Skip this operation if `None`.
    """
    def __init__(self, *ints: Integrator, index: Optional[Index] = None):
        super().__init__('assembly')
        self.ints: List[Integrator] = [] # Integrator except GroupIntegrator.
        if len(ints) == 0:
            raise ValueError("No integrators provided.")
        for integrator in ints:
            if isinstance(integrator, GroupIntegrator):
                self.ints.extend(integrator)
            elif isinstance(integrator, Integrator):
                self.ints.append(integrator)
            else:
                raise TypeError(f"Unsupported type {integrator.__class__.__name__} "
                                "found in the inputs.")

        self.set_index(index)

    def __repr__(self):
        return "GroupIntegrator[" + ", ".join([repr(i) for i in self.ints]) + "]"

    def __iter__(self):
        yield from self.ints

    def __len__(self):
        return len(self.ints)

    def __getitem__(self, index: int):
        return self.ints[index]

    def __iadd__(self, other: Integrator): # Don't create new for the grouped
        if isinstance(other, GroupIntegrator):
            self.ints.extend(other)
        elif isinstance(other, Integrator):
            self.ints.append(other)
        else:
            return NotImplemented
        return self

    def set_index(self, index: Optional[Index], /) -> None:
        if index is not None:
            self.clear(result_only=False)
            for integrator in self.ints:
                integrator.set_index(index)

        self.index = index

    def to_global_dof(self, space: _FS) -> TensorLike:
        return self.ints[0].to_global_dof(space)

    def assembly(self, space: Union[_FS, Tuple[_FS, ...]]) -> TensorLike:
        ct = self.ints[0](space)

        for int_ in self.ints[1:]:
            new_ct = int_(space)
            fdim = min(ct.ndim, new_ct.ndim)
            if ct.shape[:fdim] != new_ct.shape[:fdim]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                   f"has an incompatible shape {tuple(new_ct.shape)} "
                                   f"with the previous {tuple(ct.shape)}.")
            if new_ct.ndim > ct.ndim:
                ct = new_ct + ct[None, ...]
            elif new_ct.ndim < ct.ndim:
                ct = ct + new_ct[None, ...]
            else:
                ct = ct + new_ct

        return ct
