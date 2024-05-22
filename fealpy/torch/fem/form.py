
from typing import Sequence, overload, List, Generic, TypeVar

from torch import Tensor

from .integrator import Integrator as _I
from ..functionspace.space import FunctionSpace


_FS = TypeVar('_FS', bound=FunctionSpace)


class Form(Generic[_FS]):
    space: _FS
    dintegrators: List[_I]
    bintegrators: List[_I]

    @overload
    def add_domain_integrator(self, I: _I) -> _I: ...
    @overload
    def add_domain_integrator(self, I: Sequence[_I]) -> List[_I]: ...
    @overload
    def add_domain_integrator(self, *I: _I) -> List[_I]: ...
    def add_domain_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                self.dintegrators.extend(I)
            else:
                self.dintegrators.append(I)
        elif len(I) >= 2:
            self.dintegrators.extend(I)
        else:
            raise RuntimeError("add_domain_integrator() is called with no arguments.")

        return I

    @overload
    def add_boundary_integrator(self, I: _I) -> _I: ...
    @overload
    def add_boundary_integrator(self, I: Sequence[_I]) -> List[_I]: ...
    @overload
    def add_boundary_integrator(self, *I: _I) -> List[_I]: ...
    def add_boundary_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                self.bintegrators.extend(I)
            else:
                self.bintegrators.append(I)
        elif len(I) >= 2:
            self.bintegrators.extend(I)
        else:
            raise RuntimeError("add_boundary_integrator() is called with no arguments.")

        return I

    def number_of_domain_integrators(self) -> int:
        return len(self.dintegrators)

    def number_of_boundary_integrators(self) -> int:
        return len(self.bintegrators)
