from typing import TypeVar, Protocol, NamedTuple
from fealpy.backend import TensorLike

class Bounds(NamedTuple):
    lb: TensorLike
    ub: TensorLike

class MultiBenchmarkProtocol(Protocol):
    def evaluate(self, x: TensorLike) -> TensorLike: ...
    def get_bounds(self) -> Bounds: ...

MultiBenchmarkT = TypeVar('MultiBenchmarkT', bound=MultiBenchmarkProtocol)

DATA_TABLE = {
    1: ('dtlz1', 'DTLZ1'),
}