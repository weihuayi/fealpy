from typing import Protocol, Sequence, TypeVar
from ...backend import TensorLike


class DarcyforchheimerDataProtocol(Protocol):
    """Protocol interface for Darcy–Forchheimer PDE data components."""
    def geo_dimension(self) -> int: ...
    def domain(self) -> Sequence[float]: ...
    def pressure(self, p: TensorLike) -> TensorLike: ...
    def velocity(self, p: TensorLike) -> TensorLike: ...
    def grad_pressure(self, p: TensorLike) -> TensorLike: ...
    def f(self, p: TensorLike) -> TensorLike: ...  
    def g(self, p: TensorLike) -> TensorLike: ... 
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike: ...

DFDataT = TypeVar('DarcyforchheimerDataT', bound=DarcyforchheimerDataProtocol)

"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""

DATA_TABLE = {
    # example name: (file_name, class_name)
    1: ("exp0001", "Exp0001"),
    2: ("exp0002", "Exp0002"),
    3: ("exp0003", "Exp0003"),
    3: ("exp0003", "Exp0003"),
    4: ("exp0004", "Exp0004"),
    5: ("exp0005", "Exp0005"),
    6: ("exp0006", "Exp0006"),
    7: ("exp0007", "Exp0007"),
    8: ("exp0008", "Exp0008"),
    9: ("exp0009", "Exp0009"),
    10: ("exp0010", "Exp0010"),
}

