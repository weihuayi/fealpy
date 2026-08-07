"""Immutable controls owned by the steady collocated SIMPLE algorithm."""

from dataclasses import dataclass

from .solver_controls import (
    gradient_reconstruction_weights,
    validate_diffusion_controls,
    validate_face_flux_correction_controls,
)


def _face_interpolation(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a concrete string.")
    if value not in {"average", "linear"}:
        raise ValueError(f"{name} must be 'average' or 'linear'.")
    return value


@dataclass(frozen=True)
class SimpleDiscretizationControls:
    """Static high-accuracy discretization choices for SIMPLE."""

    pressure_gradient_method: str = "layered_lsq"
    velocity_gradient_method: str = "layered_lsq"
    gradient_layer_weights: tuple[float, float] = (1.0, 0.25)
    gradient_boundary_weight: float = 1.0
    diffusion_method: str = "over_relaxed"
    diffusion_nonorthogonal_eps: float = 0.05
    # Interpolation of cell velocity in the momentum convection operator.
    momentum_face_interpolation: str = "average"
    # Interpolation of V/a_P in the pressure Laplacian coefficient.
    pressure_response_interpolation: str = "average"
    # Interpolation used only by the Rhie--Chow pressure stabilization.
    rhie_chow_velocity_interpolation: str = "average"
    spatial_face_velocity_scheme: str = "second_order_reconstructed"
    face_flux_correction_scheme: str = "cell_anchored_quadratic"
    face_flux_quadrature_order: int = 5
    face_flux_max_stencil_layers: int = 4
    face_flux_max_condition: float = 100.0

    def __post_init__(self) -> None:
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
        for name in (
            "momentum_face_interpolation",
            "pressure_response_interpolation",
            "rhie_chow_velocity_interpolation",
        ):
            _face_interpolation(getattr(self, name), name)
        if self.spatial_face_velocity_scheme not in {
            "interpolated",
            "second_order_reconstructed",
        }:
            raise ValueError(
                "spatial_face_velocity_scheme must be 'interpolated' or "
                "'second_order_reconstructed'."
            )
        validate_face_flux_correction_controls(
            self.face_flux_correction_scheme,
            self.face_flux_quadrature_order,
            self.face_flux_max_stencil_layers,
            self.face_flux_max_condition,
        )


@dataclass(frozen=True)
class SimpleIterationControls:
    """Outer-iteration and non-orthogonal convergence controls for SIMPLE."""

    max_iterations: int = 1500
    pressure_relaxation: float = 0.3
    momentum_equation_relaxation: float = 0.9
    momentum_relative_tolerance: float = 1.0e-7
    mass_relative_tolerance: float = 1.0e-7
    momentum_nonorthogonal_max_iterations: int = 50
    momentum_nonorthogonal_rtol: float = 1.0e-4
    momentum_nonorthogonal_atol: float = 1.0e-12
    pressure_nonorthogonal_max_iterations: int = 50
    pressure_nonorthogonal_rtol: float = 1.0e-5
    pressure_nonorthogonal_atol: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if not 0.0 < self.pressure_relaxation <= 1.0:
            raise ValueError("pressure_relaxation must be in (0, 1].")
        if not 0.0 < self.momentum_equation_relaxation <= 1.0:
            raise ValueError("momentum_equation_relaxation must be in (0, 1].")
        for name in (
            "momentum_relative_tolerance",
            "mass_relative_tolerance",
            "momentum_nonorthogonal_rtol",
            "pressure_nonorthogonal_rtol",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "momentum_nonorthogonal_max_iterations",
            "pressure_nonorthogonal_max_iterations",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in (
            "momentum_nonorthogonal_atol",
            "pressure_nonorthogonal_atol",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "SimpleDiscretizationControls",
    "SimpleIterationControls",
]
