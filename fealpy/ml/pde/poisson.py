
from typing import Sequence
from .pde import PDE


class CosCos(PDE):
    def __init__(self, domain: Sequence[float]) -> None:
        super().__init__()
        self._domain = tuple(domain)
