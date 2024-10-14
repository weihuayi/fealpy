
from typing import Sequence, overload, List, Dict, Tuple, Optional, TypeVar, Generic

from ..typing import TensorLike, Size
from ..functionspace import FunctionSpace as _FS
from .integrator import Integrator

from .. import logger
from abc import ABC

_I = TypeVar('_IT', bound=Integrator)


class Form(Generic[_I], ABC):
    _spaces: Tuple[_FS, ...]
    integrators: Dict[str, Tuple[_I, ...]]
    memory: Dict[str, Tuple[TensorLike, List[TensorLike]]]
    batch_size: int
    sparse_shape: Tuple[int, ...]

    @overload
    def __init__(self, space: _FS, *, batch_size: int=0): ...
    @overload
    def __init__(self, space: Tuple[_FS, ...], *, batch_size: int=0): ...
    @overload
    def __init__(self, *space: _FS, batch_size: int=0): ...
    def __init__(self, *space, batch_size: int=0):
        if len(space) == 0:
            raise ValueError("No space is given.")
        if isinstance(space[0], Sequence):
            space = space[0]
        self._spaces = space
        self.integrators = {}
        self._cursor = 0
        self.memory = {}
        self.batch_size = batch_size

        self._values_ravel_shape = (-1,) if self.batch_size == 0 else (self.batch_size, -1)
        self.sparse_shape = self._get_sparse_shape()

    def copy(self):
        new_obj = self.__class__(self._spaces, batch_size=self.batch_size)
        new_obj.integrators.update(self.integrators)
        new_obj.memory.update(self.memory)
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
    def add_integrator(self, I: _I, *, group: str=...) -> Tuple[_I]: ...
    @overload
    def add_integrator(self, I: Sequence[_I], *, group: str=...) -> Tuple[_I, ...]: ...
    @overload
    def add_integrator(self, *I: _I, group: str=...) -> Tuple[_I, ...]: ...
    def add_integrator(self, *I, group=None):
        """Add integrator(s) to the form.

        Parameters:
            *I (Integrator): The integrator(s) to add as a new group.
                Also accepts sequence of integrators.
            group (str | None, optional): Name of the group. Defaults to None.

        Returns:
            out (Tuple[Integrator, ...]): The integrator instance(s) added.
        """
        if len(I) == 0:
            logger.info("add_integrator() is called with no arguments.")
            return tuple()

        if len(I) == 1:
            if isinstance(I[0], Sequence):
                I = tuple(I[0])
        group = f'_group_{self._cursor}' if group is None else group
        self._cursor += 1

        if group in self.integrators:
            self.integrators[group] += I
            self.clear_memory(group)
        else:
            self.integrators[group] = I

        return I

    def clear_memory(self, group: Optional[str]=None) -> None:
        """Clear the cache of the form, including global output and group output.

        Parameters:
            group (str | None, optional): The name of integrator group to clear\
            the result from. Defaults to None. Clear all cache if `None`.
        """
        if group is None:
            self.memory.clear()
        else:
            self.memory.pop(group, None)

    def _assembly_group(self, group: str, retain_ints: bool=False):
        if group in self.memory:
            return self.memory[group]

        INTS = self.integrators[group]
        ct = INTS[0](self.space)
        etg = [INTS[0].to_global_dof(s) for s in self._spaces]

        for int_ in INTS[1:]:
            new_ct = int_(self.space)
            fdim = min(ct.ndim, new_ct.ndim)
            if ct.shape[:fdim] != new_ct.shape[:fdim]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                   f"has an incompatible shape {tuple(new_ct.shape)} "
                                   f"with the previous {tuple(ct.shape)} in the group '{group}'.")
            if new_ct.ndim > ct.ndim:
                ct = new_ct + ct[None, ...]
            elif new_ct.ndim < ct.ndim:
                ct = ct + new_ct[None, ...]
            else:
                ct = ct + new_ct

        if retain_ints:
            self.memory[group] = (ct, etg)

        return ct, etg
