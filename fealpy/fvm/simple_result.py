"""Frozen result bindings returned by the steady SIMPLE solver."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from fealpy.typing import TensorLike


SimpleTerminationReason: TypeAlias = Literal[
    "fixed_point_residuals",
    "max_iterations",
]


@dataclass(frozen=True)
class SimpleIterationResidual:
    iteration: int
    mass_relative_l2: float
    mass_relative_l1: float
    mass_divergence_l2: float
    mass_imbalance_linf: float
    pressure_correction_l2: float
    momentum_absolute_l2: float
    momentum_relative_l2: float
    pressure_nonorthogonal_iterations: int
    momentum_nonorthogonal_iterations: int
    momentum_nonorthogonal_tolerance: float

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("iteration must be positive.")
        if self.pressure_nonorthogonal_iterations < 0:
            raise ValueError(
                "pressure nonorthogonal iterations must be non-negative."
            )
        if self.momentum_nonorthogonal_iterations < 0:
            raise ValueError(
                "momentum nonorthogonal iterations must be non-negative."
            )
        scalars = (
            self.mass_relative_l2,
            self.mass_relative_l1,
            self.mass_divergence_l2,
            self.mass_imbalance_linf,
            self.pressure_correction_l2,
            self.momentum_absolute_l2,
            self.momentum_relative_l2,
            self.momentum_nonorthogonal_tolerance,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("SIMPLE residual values must be finite.")


@dataclass(frozen=True)
class SimpleSolveResult:
    """One SIMPLE solve result with shape-consistent tensor bindings.

    ``frozen=True`` prevents rebinding fields; it does not make backend tensor
    payloads deeply immutable.
    """

    velocity: TensorLike
    pressure: TensorLike
    face_velocity: TensorLike
    face_flux: TensorLike
    residual_history: tuple[SimpleIterationResidual, ...]
    termination_reason: SimpleTerminationReason

    def __post_init__(self) -> None:
        if self.velocity.ndim != 2:
            raise ValueError("velocity must have shape (NC, GD).")
        NC, GD = self.velocity.shape
        if self.pressure.shape != (NC,):
            raise ValueError("pressure must have shape (NC,).")
        if (
            self.face_velocity.ndim != 2
            or self.face_velocity.shape[1] != GD
        ):
            raise ValueError("face velocity must have shape (NF, GD).")
        NF = self.face_velocity.shape[0]
        if self.face_flux.shape != (NF,):
            raise ValueError("face flux must have shape (NF,).")
        if not self.residual_history:
            raise ValueError("outer_iterations must be positive.")
        if any(
            residual.iteration != index
            for index, residual in enumerate(
                self.residual_history,
                start=1,
            )
        ):
            raise ValueError(
                "residual history iterations must be consecutive."
            )
        if self.termination_reason not in {
            "fixed_point_residuals",
            "max_iterations",
        }:
            raise ValueError("unknown SIMPLE termination reason.")

    @property
    def converged(self) -> bool:
        """Return whether the fixed-point residual criterion stopped SIMPLE."""
        return self.termination_reason == "fixed_point_residuals"

    @property
    def outer_iterations(self) -> int:
        """Return the number of recorded outer SIMPLE iterations."""
        return len(self.residual_history)


__all__ = [
    "SimpleIterationResidual",
    "SimpleSolveResult",
    "SimpleTerminationReason",
]
