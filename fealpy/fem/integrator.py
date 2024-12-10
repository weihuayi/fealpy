
from typing import Union, Optional, Any, TypeVar, Tuple, List, Dict, Sequence
from typing import overload, Generic
import logging

from .. import logger
from ..typing import TensorLike, Index, CoefLike
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
_SpaceGroup = Union[_FS, Tuple[_FS, ...]]
_OpIndex = Optional[Index]


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

    Integrators are designed to integral given functions in input spaces.
    Output of integrators are tensors on entities (0-axis), containing data
    of each local DoFs (1-axis). There may be extra dimensions.

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
    _region: Optional[TensorLike] = None

    def __init__(self, method='assembly', keep_data=False, *args, **kwds) -> None:
        if method not in self._assembly_name_map:
            raise ValueError(f"No assembly method is registered as '{method}'. "
                             f"For {self.__class__.__name__}, only the following options are available: "
                             f"{', '.join(self._assembly_name_map.keys())}.")

        self._method = method
        self._cache: Dict[Tuple[str, int], Any] = {}
        self.keep_data(keep_data)

    ### START: Cache System ###
    def keep_data(self, status_on=True, /):
        """Set whether to keep the integral material decorated by @enable_cache."""
        self._keep_data = status_on
        if not status_on:
            self._cache.clear()
        return self

    def clear(self) -> None:
        """Clear the cache of integrator."""
        self._cache.clear()
    ### END: Cache System ###

    ### START: Region of Integration ###
    def set_region(self, region: Optional[TensorLike], /):
        """Set the region of integration, given as indices of mesh entity."""
        self._region = region
        return self

    def get_region(self):
        """Get the region of integration, returned as indices of mesh entity."""
        if self._region is None:
            raise RuntimeError("Region of integration not specified. "
                               "Use Integrator.set_region to set indices.")
        return self._region
    ### END: Region of Integration ###

    def const(self, space: _SpaceGroup, /):
        value = self.assembly(space)
        to_gdof = self.to_global_dof(space)
        return ConstIntegrator(value, to_gdof)

    def __call__(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike:
        logger.debug(f"(INTEGRATOR RUN) {self.__class__.__name__}, on {space.__class__.__name__}")
        meth = getattr(self, self._assembly_name_map[self._method], None)
        if indices is None:
            val = meth(space) # Old API
        else:
            val = meth(space, indices=indices)
        if logger.level == logging._nameToLevel['INFO']:
            logger.info(f"Local tensor sized {ftype_memory_size(val)} Mb.")
        return val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._method})"

    @overload
    def to_global_dof(self, space: _FS, /, indices: _OpIndex = None) -> TensorLike: ...
    @overload
    def to_global_dof(self, space: Tuple[_FS, ...], /, indices: _OpIndex = None) -> Tuple[TensorLike, ...]: ...
    def to_global_dof(self, space: _SpaceGroup, /, indices: _OpIndex = None):
        """Return the relationship between the integral entities
        and the global dofs."""
        raise NotImplementedError

    @assemblymethod('assembly') # The default assemply method
    def assembly(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike:
        """The default method of integration on entities."""
        raise NotImplementedError

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

_GT = TypeVar('_GT')

class ConstIntegrator(Integrator, Generic[_GT]):
    """An integral item with given values.

    ConstIntegrator wrap a given TensorLike object as an Integrator type.
    The `to_gdof` is optional but must be provided if `to_global_dof` is needed.
    Indices of entity is ignored if `enable_region` is False.
    """
    def __init__(self, value: TensorLike, to_gdof: Optional[_GT] = None):
        super().__init__('assembly', False, False)
        self.value = value
        self.to_gdof = to_gdof
        self._region = slice(None)

    def set_region(self, region, /):
        logger.warning("`set_region` has no effect for ConstIntegrator.")
        return super().set_region(region)

    def to_global_dof(self, space, /, indices: _OpIndex = None) -> _GT:
        if self.to_gdof is None:
            raise RuntimeError("to_gdof not defined for ConstIntegrator.")
        if indices is None:
            return self.to_gdof
        if isinstance(self.to_gdof, (tuple, list)):
            return self.to_gdof.__class__(tg[indices] for tg in self.to_gdof)
        return self.to_gdof[indices]

    def assembly(self, space, /, indices: _OpIndex = None):
        if indices is None:
            return self.value
        return self.value[indices]


class GroupIntegrator(Integrator):
    """Combine multiple integral items as one.

    GroupIntegrator requires all sub-integrators have the same to_global_dof output.
    That is to say, all integral items must have the same domains and local-global relationship for DoFs.

    Managed by the GroupIntegrator, as a result, sub-integrators ignored their
    own to_global_dof, outputing integrals only.
    The grouped integrator takes the first integrator's `to_global_dof` as its implementation.

    Note: All sub-integrators' region will be replaced by the `region` parameter.
    Skip this operation if `None`.
    """
    def __init__(self, *ints: Integrator, region: Optional[TensorLike] = None):
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
        if region is not None:
            self.set_region(region)

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

    def set_region(self, region: TensorLike, /) -> None:
        for integrator in self.ints:
            integrator.set_region(region)
        return super().set_region(region)

    @overload
    def to_global_dof(self, space: _FS, /, indices: _OpIndex = None) -> TensorLike: ...
    @overload
    def to_global_dof(self, space: Tuple[_FS, ...], /, indices: _OpIndex = None) -> Tuple[TensorLike, ...]: ...
    def to_global_dof(self, space: _SpaceGroup, /, indices: _OpIndex = None):
        if indices is None:
            return self.ints[0].to_global_dof(space)
        return self.ints[0].to_global_dof(space, indices=indices)

    def assembly(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike:
        ct = self.ints[0](space, indices=indices)

        for int_ in self.ints[1:]:
            new_ct = int_(space, indices=indices)
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
