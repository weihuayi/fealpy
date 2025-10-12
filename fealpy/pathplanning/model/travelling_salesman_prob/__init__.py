from typing import TypeVar, Protocol
from fealpy.backend import TensorLike
from ...model import OptResult

class TravellingSalesmanProProtocol(Protocol):
    """
    Protocol defining the interface for Traveling Salesman Problem (TSP) algorithms.

    This protocol specifies the required methods for TSP implementations, including
    cost calculation, optimization execution, and solution visualization.

    Methods:
        cost_function(sol): Calculate the total cost of a TSP solution.
        opt(n): Execute the optimization algorithm.
        visualization(): Visualize the TSP solution.
    """
    def cost_function(self, sol: TensorLike) -> float: ...
    def opt(self, n: int) -> OptResult: ...
    def visualization(self) -> None: ...

TravellingSalesmanPro = TypeVar('TravellingSalesmanPro', bound=TravellingSalesmanProProtocol)

DATA_TABLE = {
    1: ('travelling_salesman_prob', 'TravellingSalesmanProb'),
    2: ('multiple_traveling_salesman_prob', 'MultipleTravelingSalesmanProb'),
}