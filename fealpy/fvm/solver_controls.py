"""Control parameter containers for collocated FVM solvers."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

from fealpy.backend import backend_manager as bm


def positive_scalar(value, name: str) -> float:
    """Return ``value`` as a positive Python scalar."""
    if callable(value):
        value = value()
    try:
        scalar = float(value)
    except TypeError:
        scalar = float(bm.to_numpy(value))
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def nonnegative_scalar(value, name: str) -> float:
    """Return ``value`` as a non-negative Python scalar."""
    if callable(value):
        value = value()
    try:
        scalar = float(value)
    except TypeError:
        scalar = float(bm.to_numpy(value))
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return scalar


def gradient_reconstruction_weights(layer_weights, boundary_weight):
    """Validate and normalize shared LSQ gradient weights."""
    try:
        if isinstance(layer_weights, (int, float)):
            layer_weights = (float(layer_weights), float(layer_weights))
        else:
            layer_weights = tuple(float(weight) for weight in layer_weights)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "gradient_layer_weights must be a scalar or a pair."
        ) from exc
    if len(layer_weights) != 2:
        raise ValueError("gradient_layer_weights must contain two values.")
    if layer_weights[0] < 0.0 or layer_weights[1] < 0.0:
        raise ValueError("gradient_layer_weights must be non-negative.")
    if layer_weights[0] == 0.0 and layer_weights[1] == 0.0:
        raise ValueError("at least one gradient_layer_weights entry must be positive.")

    try:
        boundary_weight = float(boundary_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient_boundary_weight must be a scalar.") from exc
    if boundary_weight < 0.0:
        raise ValueError("gradient_boundary_weight must be non-negative.")
    return layer_weights, boundary_weight


def validate_diffusion_controls(method: str, nonorthogonal_eps: float) -> None:
    """Validate one complete non-orthogonal diffusion variant."""
    if method not in {
        "over_relaxed",
        "bounded_over_relaxed",
        "uncorrected",
    }:
        raise ValueError(
            "diffusion_method must be 'over_relaxed', "
            "'bounded_over_relaxed', or 'uncorrected'."
        )
    if nonorthogonal_eps <= 0.0:
        raise ValueError("diffusion_nonorthogonal_eps must be positive.")


def validate_face_flux_correction_controls(
    method: str,
    quadrature_order: int,
    max_stencil_layers: int,
    max_condition: float,
) -> None:
    """Validate conservative face-average flux reconstruction controls."""
    if method not in {"none", "cell_anchored_quadratic"}:
        raise ValueError(
            "face_flux_correction_scheme must be 'none' or "
            "'cell_anchored_quadratic'."
        )
    if quadrature_order < 2:
        raise ValueError("face_flux_quadrature_order must be at least 2.")
    if max_stencil_layers < 1:
        raise ValueError("face_flux_max_stencil_layers must be positive.")
    if not math.isfinite(max_condition) or max_condition < 1.0:
        raise ValueError(
            "face_flux_max_condition must be finite and at least 1."
        )


class SolverControlsMapping:
    """Construct typed solver controls from model option mappings."""

    @classmethod
    def option_names(cls) -> frozenset[str]:
        """Return names owned by this controls object."""
        return frozenset(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls, options):
        """Return controls using only non-``None`` fields owned by ``cls``."""
        values = {
            name: options[name]
            for name in cls.option_names()
            if name in options and options[name] is not None
        }
        return cls(**values)


@dataclass(frozen=True)
class PoissonSolverControls(SolverControlsMapping):
    """Discretization and deferred-correction controls for Poisson FVM."""

    space_degree: int = 0
    gradient_method: str = "layered_lsq"
    gradient_layer_weights: tuple[float, float] = (1.0, 0.25)
    gradient_boundary_weight: float = 1.0
    diffusion_method: str = "over_relaxed"
    diffusion_nonorthogonal_eps: float = 0.05
    cross_flux_limiter: str = "none"
    cross_flux_limit_coeff: float = 0.5
    nonorthogonal_max_iter: int = 100
    nonorthogonal_rtol: float = 1.0e-10
    nonorthogonal_atol: float = 1.0e-12
    nonorthogonal_relaxation: float = 1.0

    def __post_init__(self):
        if self.space_degree != 0:
            raise ValueError("space_degree must be 0 for cell-centred FVM.")
        layer_weights, boundary_weight = gradient_reconstruction_weights(
            self.gradient_layer_weights,
            self.gradient_boundary_weight,
        )
        object.__setattr__(self, "gradient_layer_weights", layer_weights)
        object.__setattr__(self, "gradient_boundary_weight", boundary_weight)
        validate_diffusion_controls(
            self.diffusion_method,
            self.diffusion_nonorthogonal_eps,
        )
        if self.cross_flux_limiter not in {"none", "orthogonal_flux_ratio"}:
            raise ValueError(
                "cross_flux_limiter must be 'none' or 'orthogonal_flux_ratio'."
            )
        if not 0.0 <= self.cross_flux_limit_coeff <= 1.0:
            raise ValueError("cross_flux_limit_coeff must be in [0, 1].")
        if self.nonorthogonal_max_iter < 0:
            raise ValueError("nonorthogonal_max_iter must be non-negative.")
        if self.nonorthogonal_rtol < 0.0 or self.nonorthogonal_atol < 0.0:
            raise ValueError(
                "nonorthogonal_rtol and nonorthogonal_atol must be non-negative."
            )
        if self.nonorthogonal_rtol == 0.0 and self.nonorthogonal_atol == 0.0:
            raise ValueError(
                "at least one non-orthogonal residual tolerance must be positive."
            )
        if not 0.0 < self.nonorthogonal_relaxation <= 1.0:
            raise ValueError("nonorthogonal_relaxation must be in (0, 1].")


@dataclass(frozen=True)
class PisoSolverControls(SolverControlsMapping):
    """Discretization and iteration controls for ``CollocatedPisoSolver``.

    The momentum and pressure non-orthogonal counters are safety limits for
    tolerance-driven explicit correction solves.  A value of zero disables
    the explicit cross correction while still solving the base equation once.
    """

    duration: tuple[float, float] = (0.0, 1.0)
    time_steps: int = 20
    n_correctors: int = 2
    snapshot_interval: int = 1
    snapshot_start_step: int = 1
    pressure_gradient_method: str = "layered_lsq"
    velocity_gradient_method: str = "layered_lsq"
    gradient_layer_weights: tuple[float, float] = (1.0, 0.25)
    gradient_boundary_weight: float = 1.0
    diffusion_method: str = "over_relaxed"
    diffusion_nonorthogonal_eps: float = 0.05
    momentum_face_interpolation: str = "average"
    pressure_response_interpolation: str = "average"
    rhie_chow_velocity_interpolation: str = "average"
    use_transient_flux_correction: bool = True
    momentum_nonorthogonal_max_iterations: int = 50
    momentum_nonorthogonal_rtol: float = 1.0e-5
    momentum_nonorthogonal_atol: float = 1.0e-12
    pressure_nonorthogonal_max_iterations: int = 50
    pressure_nonorthogonal_rtol: float = 1.0e-5
    pressure_nonorthogonal_atol: float = 1.0e-12
    diagnostics_enabled: bool = False

    def __post_init__(self):
        layer_weights, boundary_weight = gradient_reconstruction_weights(
            self.gradient_layer_weights,
            self.gradient_boundary_weight,
        )
        object.__setattr__(self, "gradient_layer_weights", layer_weights)
        object.__setattr__(self, "gradient_boundary_weight", boundary_weight)
        validate_diffusion_controls(
            self.diffusion_method,
            self.diffusion_nonorthogonal_eps,
        )
        self.validate_time_controls(self.duration, self.time_steps)
        self.validate_piso_controls(self.n_correctors)
        self.validate_snapshot_controls(self.snapshot_interval, self.snapshot_start_step)
        for name in (
            "momentum_face_interpolation",
            "pressure_response_interpolation",
            "rhie_chow_velocity_interpolation",
        ):
            self.validate_face_interpolation(
                getattr(self, name),
                name,
            )
        self.validate_nonorthogonal_controls(
            self.momentum_nonorthogonal_max_iterations,
            self.momentum_nonorthogonal_rtol,
            self.momentum_nonorthogonal_atol,
            self.pressure_nonorthogonal_max_iterations,
            self.pressure_nonorthogonal_rtol,
            self.pressure_nonorthogonal_atol,
        )

    @property
    def tau(self) -> float:
        """Return the uniform time-step size."""
        return (self.duration[1] - self.duration[0]) / self.time_steps

    @staticmethod
    def validate_time_controls(duration, time_steps: int) -> None:
        if time_steps < 1:
            raise ValueError("time_steps must be positive.")
        if len(duration) != 2 or duration[1] <= duration[0]:
            raise ValueError("duration must be an increasing pair.")

    @staticmethod
    def validate_piso_controls(n_correctors: int) -> None:
        if n_correctors < 1:
            raise ValueError("n_correctors must be positive.")

    @staticmethod
    def validate_snapshot_controls(interval: int, start_step: int) -> None:
        if interval < 1:
            raise ValueError("snapshot_interval must be positive.")
        if start_step < 1:
            raise ValueError("snapshot_start_step must be positive.")

    @staticmethod
    def validate_face_interpolation(method: str, name: str) -> str:
        if method not in {"average", "linear"}:
            raise ValueError(
                f"{name} must be 'average' or 'linear'."
            )
        return method

    @staticmethod
    def validate_nonorthogonal_controls(
        momentum_max_iter: int,
        momentum_tol: float,
        momentum_atol: float,
        pressure_max_iter: int,
        pressure_tol: float,
        pressure_atol: float,
    ) -> None:
        if momentum_max_iter < 0:
            raise ValueError(
                "momentum_nonorthogonal_max_iterations must be non-negative."
            )
        if pressure_max_iter < 0:
            raise ValueError(
                "pressure_nonorthogonal_max_iterations must be non-negative."
            )
        if momentum_tol <= 0.0:
            raise ValueError("momentum_nonorthogonal_rtol must be positive.")
        if momentum_atol < 0.0:
            raise ValueError("momentum_nonorthogonal_atol must be non-negative.")
        if pressure_tol <= 0.0:
            raise ValueError("pressure_nonorthogonal_rtol must be positive.")
        if pressure_atol < 0.0:
            raise ValueError("pressure_nonorthogonal_atol must be non-negative.")
