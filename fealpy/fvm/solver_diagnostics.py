"""Reusable scalar diagnostics for FVM solver loops."""

import logging
from dataclasses import dataclass
from typing import Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .simple_result import SimpleIterationResidual


@dataclass(frozen=True)
class EquationResidual:
    """Norms of one complete discrete equation balance."""

    absolute: float
    relative: float
    scale: float


def normalized_equation_residual(
    lhs: TensorLike,
    rhs: TensorLike,
    eps: float = 1.0e-30,
    *,
    normalization_lhs: TensorLike | None = None,
    normalization_rhs: TensorLike | None = None,
    norm_weights: TensorLike | None = None,
    scale_mode: Literal["sum", "max"] = "sum",
) -> EquationResidual:
    """Return norms of the complete discrete balance ``lhs - rhs``.

    The residual vector always uses ``lhs - rhs``.  By default its relative
    value is normalized by ``||lhs|| + ||rhs||``.  Callers solving an
    algebraically relaxed equation may instead provide the corresponding
    unrelaxed physical balance through ``normalization_lhs`` and
    ``normalization_rhs``.  ``scale_mode="max"`` matches the physical SIMPLE
    momentum residual convention without changing pressure/Poisson semantics.
    """
    if (normalization_lhs is None) != (normalization_rhs is None):
        raise ValueError(
            "normalization_lhs and normalization_rhs must be provided together."
        )
    def residual_norm(value):
        if norm_weights is None:
            return bm.linalg.norm(value)
        return bm.sqrt(bm.sum(norm_weights * value**2))

    residual = lhs - rhs
    absolute = float(bm.to_numpy(residual_norm(residual)))
    scale_lhs = lhs if normalization_lhs is None else normalization_lhs
    scale_rhs = rhs if normalization_rhs is None else normalization_rhs
    lhs_norm = float(bm.to_numpy(residual_norm(scale_lhs)))
    rhs_norm = float(bm.to_numpy(residual_norm(scale_rhs)))
    if scale_mode == "sum":
        scale = lhs_norm + rhs_norm
    elif scale_mode == "max":
        scale = max(lhs_norm, rhs_norm)
    else:
        raise ValueError("scale_mode must be 'sum' or 'max'.")
    return EquationResidual(
        absolute=absolute,
        relative=absolute / max(scale, float(eps)),
        scale=scale,
    )


def normalized_relaxed_equation_residual(
    lhs: TensorLike,
    rhs: TensorLike,
    solution: TensorLike,
    previous_solution: TensorLike,
    relaxation_diagonal: TensorLike,
    eps: float = 1.0e-30,
    *,
    norm_weights: TensorLike | None = None,
) -> EquationResidual:
    """Normalize a relaxed-system defect by its unrelaxed physical balance.

    For ``A_alpha = A + Delta`` and
    ``b_alpha = b + Delta * x_old``, ``lhs`` and ``rhs`` are the relaxed
    balance ``A_alpha x`` and ``b_alpha + c(x)``.  The residual vector remains
    their exact difference, while the normalization references ``A x`` and
    ``b + c(x)``.
    """
    physical_lhs = lhs - relaxation_diagonal * solution
    physical_rhs = rhs - relaxation_diagonal * previous_solution
    return normalized_equation_residual(
        lhs,
        rhs,
        eps,
        normalization_lhs=physical_lhs,
        normalization_rhs=physical_rhs,
        norm_weights=norm_weights,
        scale_mode="max",
    )


def inexact_inner_tolerance(
    configured_tolerance: float,
    target_tolerance: float,
    outer_residual: float | None = None,
    *,
    forcing_factor: float = 0.1,
) -> float:
    """Return a residual-driven tolerance for an inexact inner solve.

    ``configured_tolerance`` is the loose early-iteration limit and
    ``target_tolerance`` is the accuracy required at the outer fixed point.
    Once an outer residual is available, the inner tolerance follows a simple
    forcing sequence and is kept between those two limits.
    """
    configured_tolerance = float(configured_tolerance)
    target_tolerance = float(target_tolerance)
    forcing_factor = float(forcing_factor)
    if configured_tolerance <= 0.0:
        raise ValueError("configured_tolerance must be positive.")
    if target_tolerance <= 0.0:
        raise ValueError("target_tolerance must be positive.")
    if forcing_factor <= 0.0:
        raise ValueError("forcing_factor must be positive.")
    if outer_residual is None:
        return configured_tolerance
    outer_residual = float(outer_residual)
    if outer_residual < 0.0:
        raise ValueError("outer_residual must be non-negative.")

    strict_tolerance = min(configured_tolerance, target_tolerance)
    forced_tolerance = forcing_factor * outer_residual
    return max(
        strict_tolerance,
        min(configured_tolerance, forced_tolerance),
    )


def equation_residual_converged(
    metrics: EquationResidual,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Return whether a complete equation residual meets mixed tolerances."""
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")
    if rtol == 0.0 and atol == 0.0:
        raise ValueError("at least one of rtol or atol must be positive.")
    return metrics.absolute <= atol + rtol * metrics.scale


def linf_norm(value: TensorLike) -> float:
    """Return the infinity norm as a Python float."""
    return float(bm.to_numpy(bm.max(bm.abs(value))))


def format_pressure_correction_log(
    *,
    iteration: int,
    nonorthogonal_iterations: int,
    pressure_criterion: float,
    momentum_residual: float,
    mass_residual: float,
    pressure_correction: float,
    label: str = "SIMPLE",
) -> str:
    """Format one pressure-correction iteration diagnostic line."""
    return (
        f"[{label} {iteration}] "
        f"nonorthogonal iterations: {nonorthogonal_iterations}, "
        f"pressure criterion: {pressure_criterion:.2e}, "
        f"momentum residual: {momentum_residual:.2e}, "
        f"mass residual: {mass_residual:.2e}, "
        f"pressure correction L2: {pressure_correction:.2e}"
    )


def simple_iteration_log_message(
    *,
    simple_iteration: int,
    nonorthogonal_iterations: int,
    pressure_criterion: float,
    momentum_residual: float,
    mass_residual: float,
    pressure_correction: float,
) -> str:
    """Format one SIMPLE iteration diagnostic line."""
    return format_pressure_correction_log(
        iteration=simple_iteration,
        nonorthogonal_iterations=nonorthogonal_iterations,
        pressure_criterion=pressure_criterion,
        momentum_residual=momentum_residual,
        mass_residual=mass_residual,
        pressure_correction=pressure_correction,
        label="SIMPLE",
    )


def log_simple_iteration(
    logger: logging.Logger,
    iteration: int,
    residual: SimpleIterationResidual,
) -> None:
    """Log one SIMPLE pressure-correction diagnostic record."""
    logger.info(
        simple_iteration_log_message(
            simple_iteration=iteration,
            nonorthogonal_iterations=(
                residual.pressure_nonorthogonal_iterations
            ),
            pressure_criterion=residual.pressure_correction_l2,
            momentum_residual=residual.momentum_relative_l2,
            mass_residual=residual.mass_relative_l2,
            pressure_correction=residual.pressure_correction_l2,
        )
    )
