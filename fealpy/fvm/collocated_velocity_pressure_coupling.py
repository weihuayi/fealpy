"""Cell velocity-pressure coupling for collocated FVM solvers."""

from fealpy.typing import TensorLike


def correct_cell_velocity(
    cell_velocity: TensorLike,
    pressure_gradient: TensorLike,
    cell_response: TensorLike,
) -> TensorLike:
    """Apply ``U <- U - (V/a_P) grad(p)``."""
    return cell_velocity - cell_response[:, None] * pressure_gradient


def remove_pressure_response(
    cell_velocity: TensorLike,
    pressure_gradient: TensorLike,
    cell_response: TensorLike,
) -> TensorLike:
    """Return ``U + (V/a_P) grad(p)`` for a pressure-free state."""
    return cell_velocity + cell_response[:, None] * pressure_gradient


__all__ = [
    "correct_cell_velocity",
    "remove_pressure_response",
]
