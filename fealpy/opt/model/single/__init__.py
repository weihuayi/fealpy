from typing import TypeVar, Protocol
from fealpy.backend import TensorLike

class SingleBenchmarkProtocol(Protocol):
    """
    Protocol defining the interface for single-objective benchmark optimization problems.

    This protocol specifies the required method that any single-objective benchmark
    optimization problem must implement for objective function evaluation.

    Methods:
        evaluate(x): Evaluate the objective function for given input values.
    """
    def evaluate(self, x: TensorLike) -> TensorLike: ...

SingleBenchmarkT = TypeVar('SingleBenchmarkT', bound=SingleBenchmarkProtocol)

DATA_TABLE = {
    1: ('cec', 'CEC2017'),
    2: ('cec', 'CEC2020'),
    3: ('cec', 'CEC2022'),
}