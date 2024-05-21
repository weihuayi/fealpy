
from typing import Sequence

from torch import Tensor

from .integrator import Integrator


class IntegratorHandler():
    def __init__(self, integrator: Integrator, form=None) -> None:
        self.integrator = integrator
        self.form = form
        self._value = None

    @classmethod
    def build_list(cls, integrators: Sequence[Integrator], form=None):
        return [cls(integrator, form) for integrator in integrators]

    @property
    def value(self) -> Tensor:
        if self._value is None:
            self._value = self.integrator.assembly()
        return self._value

    def assembly(self) -> Tensor:
        return self.integrator.assembly()

    def clear(self) -> None:
        self._value = None
