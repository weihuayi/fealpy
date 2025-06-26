
from typing import Union, Optional, Any, TypeVar, Tuple, List, Dict, Callable
from typing import Generic
import logging
from ..typing import TensorLike, Index, CoefLike
from .. import logger
from ..mesh import Mesh
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
_Region = Union[Callable[[Mesh], TensorLike], TensorLike, None]


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
            def assembly(self, space: _FS, /, indices=None) -> Tensor:
                # 'assembly' is the default assembly method,
                # naturally registered to name 'assembly'.
                return integral

            @assemblymethod('my')
            def my_assembly(self, space: _FS, /, indices=None) -> Tensor:
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
    def wrapper(integrator_obj, space, /, indices=None) -> TensorLike:
        if (indices is None) and (integrator_obj._keep_data):
            assert hasattr(integrator_obj, '_cache')
            _cache = integrator_obj._cache
            key = (func.__name__, id(space))

            if key in _cache:
                return _cache[key]

            data = func(integrator_obj, space)
            _cache[key] = data

            return data
        else:
            if indices is None:
                return func(integrator_obj, space)
            return func(integrator_obj, space, indices)

    return wrapper


class Integrator(metaclass=IntegratorMeta):
    """The base class for integrators.

    ## Introduction

    Integrators are designed to integral given functions in input spaces.
    Output of integrators are tensors on entities (0-axis), containing data
    of each local DoFs (1-axis). There may be extra dimensions.

    Integrators have a concept called `region` to specify the region of integration,
    given as indices of mesh entities.
    Integrators are expected to output tensor fields on these mesh entities
    (i.e. tensors sized the number of entity in the 0-dimension).

    All integrators should implement methods named `assembly` and `to_global_dof`.
    See examples below:
    ```
    def to_global_dof(space, /, indices=None):
        pass

    def assembly(space, /, indices=None):
        pass
    ```
    These two methods indicate two functions of an integrator: calculation of the integral,
    and getting relationship between local DoFs and global DoFs, repectively.
    The `indices` argument is designed to select a subset of integrator's working region,
    and the final indices of entities can be fetched by
    ```
    # inside the methods of integrators
    index = self.entity_selection(indices)
    ```
    Users can customize an integrator by implementing them in a subclass.

    ## Features

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
    _region: _Region = None
    etype: str

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
    def set_region(self, region: _Region, /):
        """Set the region of integration, given as indices of mesh entity,
        or a callable that receives a mesh and returns the indices."""
        self._region = region
        self.clear()
        return self

    def get_region(self):
        """Get the region of integration, returned as indices of mesh entity,
        or a callable that receives a mesh and returns the indices."""
        return self._region

    def entity_selection(self, indices: _OpIndex = None, *, mesh: Optional[Mesh] = None) -> Index:
        """Make the selection of integral entities."""
        if self._region is None:
            if indices is None:
                return slice(None, None, None)
            else:
                return indices
        else:
            if callable(self._region):
                if mesh is None:
                    raise RuntimeError("Mesh must be provided in entity_selection "
                    "when region is given as a callable.")
                full_region = self._region(mesh)
            else:
                full_region = self._region
            if indices is None:
                return full_region
            else:
                if bm.is_tensor(full_region):
                    if full_region.dtype == bm.bool:
                        return bm.nonzero(full_region)[0][indices]
                    return full_region[indices]
                else:
                    raise TypeError(f"region of type '{full_region.__class__.__name__}' "
                                    "is not supported when indices is given.")

    def size(self, mesh: Mesh, /) -> int:
        if self._region is None:
            if not hasattr(self, 'etype'):
                raise RuntimeError("etype of Integrator should be specified to detect "
                "the number of entities when region is `None`.")
            else:
                return mesh.count(self.etype)
        else:
            if callable(self._region):
                full_region = self._region(mesh)
            else:
                full_region = self._region
            if bm.is_tensor(full_region):
                if full_region.dtype == bm.bool:
                    return bm.sum(full_region, dtype=bm.int64)
                else:
                    return full_region.shape[0]
            else:
                raise TypeError(f"region of type '{full_region.__class__.__name__}' "
                                "is not supported when indices is given.")
    ### END: Region of Integration ###

    def const(self, space: _SpaceGroup, /):
        value = self(space)
        to_gdof = self.to_global_dof(space)
        return ConstIntegrator(value, to_gdof)

    def __call__(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike:
        logger.debug(f"(INTEGRATOR RUN) {self.__class__.__name__}, on {space.__class__.__name__}")
        meth = getattr(self, self._assembly_name_map[self._method], None)
        if indices is None:
            val = meth(space) # Old API
        else:
            val = meth(space, indices=indices)
        if logger.level <= logging._nameToLevel['INFO']:
            logger.info(f"Local tensor sized {ftype_memory_size(val)} Mb.")
        return val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._method})"

    def to_global_dof(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> Union[TensorLike, Tuple[TensorLike, ...]]:
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
    etype = 'cell'

class FaceInt(Integrator):
    """### Face Integrator
    Base class for integrators that integrate over mesh faces."""
    etype = 'face'

class EdgeInt(Integrator):
    """### Edge Integrator
    Base class for integrators that integrate over mesh edges."""
    etype = 'edge'


##################################################
### Integral Utils
##################################################

_GT = TypeVar('_GT')

class ConstIntegrator(Integrator, Generic[_GT]):
    """An integral item with given values.

    ConstIntegrator wrap a given TensorLike object as an Integrator type.
    The `to_gdof` is optional but must be provided if `to_global_dof` is needed.
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

    def to_global_dof(self, space: _SpaceGroup, /, indices: _OpIndex = None):
        if indices is None:
            return self.ints[0].to_global_dof(space)
        return self.ints[0].to_global_dof(space, indices=indices)

    def assembly(self, space: _SpaceGroup, /, indices: _OpIndex = None) -> TensorLike:
        ct = self.ints[0](space, indices=indices)
        f = lambda x: x[0] + x[1]

        for int_ in self.ints[1:]:
            new_ct = int_(space, indices=indices)
            fdim = min(ct[0].ndim, new_ct[0].ndim)
            if ct[0].shape[:fdim] != new_ct[0].shape[:fdim]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                   f"has an incompatible shape {tuple(new_ct.shape)} "
                                   f"with the previous {tuple(ct.shape)}.")
            if new_ct[0].ndim > ct[0].ndim:
                ct = list(map(f, zip(new_ct ,ct[None, ...])))
            elif new_ct[0].ndim < ct[0].ndim:
                ct = list(map(f, zip(ct ,new_ct[None, ...])))
            else:
                ct = list(map(f, zip(ct, new_ct)))

        return ct
