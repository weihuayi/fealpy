
from typing import Sequence, overload, Iterable, Dict, Tuple, Optional, Union, TypeVar, Generic

from ..typing import TensorLike, Size, Index
from ..backend import backend_manager as bm
from ..functionspace import FunctionSpace as _FS
from .integrator import Integrator, GroupIntegrator

from .. import logger
from abc import ABC

_I = TypeVar('_I', bound=Integrator)
Self = TypeVar('Self')


class Form(Generic[_I], ABC):
    _spaces: Tuple[_FS, ...]
    integrators: Dict[str, _I]
    chunk_sizes: Dict[str, int]
    batch_size: int
    sparse_shape: Tuple[int, ...]

    @overload
    def __init__(self, space: _FS, /, *, batch_size: int=0): ...
    @overload
    def __init__(self, space: Tuple[_FS, ...], /, *, batch_size: int=0): ...
    @overload
    def __init__(self, *space: _FS, batch_size: int=0): ...
    def __init__(self, *space, batch_size: int=0):
        if len(space) == 0:
            raise ValueError("No space is given.")
        if isinstance(space[0], Sequence):
            space = space[0]
        self._spaces = space
        self.integrators = {}
        self.chunk_sizes = {}
        self._cursor = 0
        self.batch_size = batch_size

        self._values_ravel_shape = (-1,) if self.batch_size == 0 else (self.batch_size, -1)
        self.sparse_shape = self._get_sparse_shape()

    def copy(self):
        new_obj = self.__class__(self._spaces, batch_size=self.batch_size)
        new_obj.integrators.update(self.integrators)
        new_obj.chunk_sizes.update(self.chunk_sizes)
        new_obj._values_ravel_shape = self._values_ravel_shape
        new_obj.sparse_shape = tuple(reversed(self.sparse_shape))
        return new_obj

    def __len__(self) -> int:
        return len(self.integrators)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self._spaces}"

    def _get_sparse_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError('Please implement the _get_sparse_shape method '
                                  'to generate the shape of the form.')

    @property
    def shape(self) -> Size:
        if self.batch_size == 0:
            return self.sparse_shape
        return (self.batch_size,) + self.sparse_shape

    @property
    def space(self):
        if len(self._spaces) == 1:
            return self._spaces[0]
        else:
            return self._spaces

    @overload
    def add_integrator(self: Self, I: _I, /, *, region: Optional[Index] = None, chunk_size: int = 0, group: str = ...) -> Self: ...
    @overload
    def add_integrator(self: Self, I: Sequence[_I], /, *, region: Optional[Index] = None, chunk_size: int = 0, group: str = ...) -> Self: ...
    @overload
    def add_integrator(self: Self, *I: _I, region: Optional[Index] = None, chunk_size: int = 0, group: str = ...) -> Self: ...
    def add_integrator(self, *I, region: Optional[Index] = None, chunk_size=0, group=None):
        """Add integrator(s) to the form.

        Parameters:
            *I (Integrator): The integrator(s) to add as a new group.
                Also accepts sequence of integrators.
            index (Index | None, optional):
            chunk_size (int, optional):
            group (str | None, optional): Name of the group. Defaults to None.

        Returns:
            out (Tuple[Integrator, ...]): The integrator instance(s) added.
        """
        if len(I) == 0:
            logger.info("add_integrator() is called with no arguments.")
            return self

        if len(I) == 1 and isinstance(I[0], Sequence):
            I = tuple(I[0])

        if len(I) == 1:
            I = I[0]
            if region is not None:
                I.set_region(region)
        else:
            I = GroupIntegrator(*I, region=region)

        return self._add_integrator_impl(I, group, chunk_size)

    @overload
    def __lshift__(self: Self, other: Integrator) -> Self: ...
    def __lshift__(self, other):
        if isinstance(other, Integrator):
            return self._add_integrator_impl(other, None)
        else:
            return NotImplemented

    def _add_integrator_impl(self, I: _I, group: Optional[str] = None, chunk_size: int = 0):
        group = f'_group_{self._cursor}' if group is None else group
        self._cursor += 1
        self.chunk_sizes[group] = chunk_size

        if group in self.integrators:
            self.integrators[group] += I
        else:
            self.integrators[group] = I

        return self

    def _assembly_group(self, group: str, /, *args, **kwds):
        integrator = self.integrators[group]
        etg = integrator.to_global_dof(self.space)
        if not isinstance(etg, (tuple, list)):
            etg = (etg, )
        return integrator(self.space), etg

    def assembly_local_iterative(self):
        """Assembly local matrix considering chunk size.
        Yields local matrix and to_global_dof tuple."""
        for key, int_ in self.integrators.items():
            chunk_size = self.chunk_sizes[key]
            if chunk_size == 0:
                logger.debug(f"(ASSEMBLY LOCAL FULL) {key}")
                yield self._assembly_group(key)
            else:
                logger.debug(f"(ASSEMBLY LOCAL ITER) {key}, {chunk_size} for each chunk")
                yield from IntegralIter.split(int_, chunk_size)(self.space)


# An iteration util for the `_assembly_group` method.
class IntegralIter():
    def __init__(self, integrator: Integrator, /, indices_or_segments: Union[Iterable[TensorLike], TensorLike]):
        self.integrator = integrator
        self.indices_or_segments = indices_or_segments

    def kernel(self, space: Union[_FS, Tuple[_FS, ...]], /, indices: Index):
        etg = self.integrator.to_global_dof(space, indices=indices)
        if not isinstance(etg, (tuple, list)):
            etg = (etg, )
        return self.integrator(space, indices=indices), etg

    def __call__(self, spaces: Tuple[_FS, ...]):
        if isinstance(self.indices_or_segments, TensorLike):
            return self._call_impl_segments(spaces, self.indices_or_segments)
        elif isinstance(self.indices_or_segments, Iterable):
            return self._call_impl_indices(spaces, self.indices_or_segments)
        else:
            raise TypeError(f"Unsupported indices or segments.")

    def _call_impl_indices(self, spaces: Tuple[_FS, ...], /, indices: Iterable[TensorLike]):
        for index in indices:
            yield self.kernel(spaces, index)

    def _call_impl_segments(self, spaces: Tuple[_FS, ...], /, segments: TensorLike):
        assert segments.ndim == 1
        start = 0
        stop = 0
        length = segments.shape[0] + 1

        for i in range(length):
            logger.debug(f"(FORM ITER) {i+1}/{length}")
            stop = segments[i] if (i + 1 < length) else None
            slicing = slice(start, stop, 1)
            yield self.kernel(spaces, slicing)
            start = stop

    @classmethod
    def split(cls, integrator: Integrator, /, chunk_size=0):
        size = integrator.get_region().shape[0]
        if chunk_size >= size:
            segments = bm.empty((0,), dtype=bm.int64)
        else:
            segments = bm.arange(chunk_size, size, chunk_size, dtype=bm.int64)
        return cls(integrator, segments)
