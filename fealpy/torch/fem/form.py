
from typing import Sequence, overload, List, Generic, TypeVar

from torch import Tensor

from .integrator import Integrator as _I
from ..functionspace.space import FunctionSpace


_FS = TypeVar('_FS', bound=FunctionSpace)


class IntegratorHandler():
    def __init__(self, integrator: _I, form=None) -> None:
        self.integrator = integrator
        self.form = form
        self._value = None

    def __getattr__(self, name: str):
        return getattr(self.integrator, name)

    @classmethod
    def build_list(cls, integrators: Sequence[_I], form=None):
        return [cls(integrator, form) for integrator in integrators]

    @property
    def value(self) -> Tensor:
        if self._value is None:
            self._value = self.integrator.assembly()
        return self._value

    def clear(self) -> None:
        self._value = None


class Form(Generic[_FS]):
    space: _FS
    dintegrators: List[IntegratorHandler]
    bintegrators: List[IntegratorHandler]

    @overload
    def add_domain_integrator(self, I: _I) -> IntegratorHandler: ...
    @overload
    def add_domain_integrator(self, I: Sequence[_I]) -> List[IntegratorHandler]: ...
    @overload
    def add_domain_integrator(self, *I: _I) -> List[IntegratorHandler]: ...
    def add_domain_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                handler = IntegratorHandler.build_list(I, form=self)
                self.dintegrators.extend(handler)
            else:
                handler = IntegratorHandler(I, form=self)
                self.dintegrators.append(handler)
        elif len(I) >= 2:
            handler = IntegratorHandler.build_list(I, form=self)
            self.dintegrators.extend(handler)
        else:
            raise RuntimeError("add_domain_integrator() is called with no arguments.")

        return handler

    @overload
    def add_boundary_integrator(self, I: _I) -> IntegratorHandler: ...
    @overload
    def add_boundary_integrator(self, I: Sequence[_I]) -> List[IntegratorHandler]: ...
    @overload
    def add_boundary_integrator(self, *I: _I) -> List[IntegratorHandler]: ...
    def add_boundary_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                handler = IntegratorHandler.build_list(I, form=self)
                self.bintegrators.extend(handler)
            else:
                handler = IntegratorHandler(I, form=self)
                self.bintegrators.append(handler)
        elif len(I) >= 2:
            handler = IntegratorHandler.build_list(I, form=self)
            self.bintegrators.extend(handler)
        else:
            raise RuntimeError("add_boundary_integrator() is called with no arguments.")

        return handler

    def number_of_domain_integrators(self) -> int:
        return len(self.dintegrators)

    def number_of_boundary_integrators(self) -> int:
        return len(self.bintegrators)
