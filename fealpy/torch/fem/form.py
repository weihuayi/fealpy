
from typing import Sequence, overload, List, Generic, TypeVar, Dict, Tuple, Optional

from torch import Tensor

from .integrator import Integrator as _I
from ..functionspace.space import FunctionSpace
from ..import logger


_FS = TypeVar('_FS', bound=FunctionSpace)


class Form(Generic[_FS]):
    space: _FS
    integrators: Dict[str, Tuple[_I, ...]]
    memory: Dict[str, Tuple[Tensor, Tensor]]
    batch_size: int

    def __len__(self) -> int:
        return len(self.integrators)

    @overload
    def add_integrator(self, I: _I, *, group: str=...) -> _I: ...
    @overload
    def add_integrator(self, I: Sequence[_I], *, group: str=...) -> List[_I]: ...
    @overload
    def add_integrator(self, *I: _I, group: str=...) -> List[_I]: ...
    def add_integrator(self, *I, group=None):
        if len(I) == 0:
            logger.info("add_integrator() is called with no arguments.")
            return None

        if len(I) == 1:
            if isinstance(I[0], Sequence):
                I = tuple(I[0])
        group = f'_group_{len(self)}' if group is None else group

        if group in self.integrators:
            self.integrators[group] += I
            self.clear_memory(group)
        else:
            self.integrators[group] = I

        return I

    def clear_memory(self, group: Optional[str]=None) -> None:
        if group is None:
            self.memory.clear()
        else:
            self.memory.pop(group, None)

    def assembly_group(self, group: str, retain_ints: bool=False):
        if group in self.memory:
            return self.memory[group]

        INTS = self.integrators[group]
        ct = INTS[0](self.space)
        etg = INTS[0].to_global_dof(self.space)

        if not retain_ints:
            INTS[0].clear()

        for int_ in INTS[1:]:
            new_ct = int_(self.space)
            fdim = min(ct.ndim, new_ct.ndim)
            if ct.shape[:fdim] != new_ct.shape[:fdim]:
                raise RuntimeError(f"The output of the integrator {int_.__class__.__name__} "
                                   f"has an incompatible shape {tuple(new_ct.shape)} "
                                   f"with the previous {tuple(ct.shape)} in the group '{group}'.")
            if new_ct.ndim > ct.ndim:
                ct = new_ct + ct.unsqueeze(-1)
            elif new_ct.ndim < ct.ndim:
                ct = ct + new_ct.unsqueeze(-1)
            else:
                ct = ct + new_ct
            if not retain_ints:
                int_.clear()

        if retain_ints:
            self.memory[group] = (ct, etg)

        return ct, etg
