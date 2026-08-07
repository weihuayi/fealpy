"""Frozen result bindings returned by the transient PISO solver."""

from dataclasses import dataclass

from fealpy.typing import TensorLike


@dataclass(frozen=True)
class PisoPressureCorrectionDiagnostics:
    """Scalar metrics produced only when one PISO corrector is diagnosed."""

    pressure_free_divergence_linf: float
    pressure_corrected_divergence_linf: float
    pressure_free_flux_linf: float
    pressure_corrected_flux_linf: float
    pressure_flux_linf: float
    pressure_orthogonal_flux_linf: float
    pressure_cross_flux_linf: float
    pressure_boundary_flux_linf: float
    transient_flux_correction_linf: float
    pressure_nonorthogonal_iterations: int


@dataclass(frozen=True)
class PisoPressureCorrectionStepResult:
    """Fixed-shape result of one PISO pressure-correction step."""

    velocity: TensorLike
    pressure: TensorLike
    face_flux: TensorLike
    face_response_coefficient: TensorLike
    diagnostics: PisoPressureCorrectionDiagnostics | None


@dataclass(frozen=True)
class PisoCorrectorDiagnostic:
    """One complete, typed PISO corrector diagnostic record."""

    step: int
    time: float
    corrector: int
    n_correctors: int
    pressure_correction: PisoPressureCorrectionDiagnostics
    operator_splitting_compensation_linf: float
    velocity_update_linf: float
    pressure_update_linf: float
    rhie_chow_flux_error_linf: float


@dataclass(frozen=True)
class PisoSnapshot:
    """One call-local transient state passed to a snapshot consumer."""

    step: int
    time: float
    velocity: TensorLike
    face_velocity: TensorLike
    pressure: TensorLike
    face_flux: TensorLike


@dataclass(frozen=True)
class PisoSolveResult:
    """One PISO solve result with shape-consistent tensor bindings."""

    velocity: TensorLike
    pressure: TensorLike
    face_velocity: TensorLike
    face_flux: TensorLike
    corrector_diagnostics: tuple[PisoCorrectorDiagnostic, ...]

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
            raise ValueError("face_velocity must have shape (NF, GD).")
        NF = self.face_velocity.shape[0]
        if self.face_flux.shape != (NF,):
            raise ValueError("face_flux must have shape (NF,).")


__all__ = [
    "PisoCorrectorDiagnostic",
    "PisoPressureCorrectionDiagnostics",
    "PisoPressureCorrectionStepResult",
    "PisoSnapshot",
    "PisoSolveResult",
]
