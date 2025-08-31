from typing import TypeVar, Protocol, NamedTuple
from fealpy.backend import TensorLike

class Bounds(NamedTuple):
    lb: TensorLike
    ub: TensorLike

class ConstrainedBenchmarkProtocol(Protocol):
    """
    Protocol defining the interface for constrained benchmark optimization problems.

    This protocol specifies the required methods that any constrained benchmark
    optimization problem must implement, including dimension query, bounds retrieval,
    penalty calculation, objective evaluation, and optimal value reference.

    Methods:
        get_dim(): Get the dimensionality of the problem.
        get_bounds(): Get the lower and upper bounds for variables.
        penalty(value): Calculate penalty for constraint violations.
        evaluate(x): Evaluate the objective function with penalties.
        get_optimal(): Get the known optimal value for validation.
    """
    def get_dim(self) -> int: ...
    def get_bounds(self) -> tuple: ...
    def penalty(self, value: TensorLike) -> TensorLike: ...
    def evaluate(self, x: TensorLike) -> TensorLike: ...
    def get_optimal(self) -> float: ...

ConstrainedBenchmarkT = TypeVar('ConstrainedBenchmarkT', bound=ConstrainedBenchmarkProtocol)

DATA_TABLE = {
    1: ('tension_spring_design', 'TensionSpringDesign'),
    2: ('pressure_vessel_design', 'PressureVesselDesign'),
    3: ('speed_reducer_design', 'SpeedReducerDesign')
}