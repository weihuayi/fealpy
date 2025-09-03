from typing import TypeVar, Protocol
from fealpy.backend import TensorLike
from ...model import OptResult

class RoutePlanningProtocol(Protocol):
    """
    Protocol defining the interface for route planning algorithms.

    This protocol specifies the required methods that any route planning implementation
    must provide, including model visualization, solution output, solver execution,
    cost calculation, and geometric utilities.

    Methods:
        plot_model(): Visualize the planning model.
        output_solution(sol, smooth): Output the solution with optional smoothing.
        solver(n): Execute the solver with specified parameters.
        cost_function(sol): Calculate the cost of a solution.
        dist_point_to_segment(P, A, B): Compute distance from point to line segment.
        prepare_uav_bounds(angle_range): Prepare UAV boundary constraints.
    """
    def visualization(self, sol: TensorLike, smooth: int) -> None: ...
    def solver(self, n: int) -> OptResult: ...
    def cost_function(self, sol: TensorLike) -> float: ...

RoutePlanningDataT = TypeVar('RoutePlanningDataT', bound=RoutePlanningProtocol)

DATA_TABLE = {
    1: ('uav_3d', 'UAV3D'),
    2: ('agv_2d', 'AGV2D'),
}