"""Named, reproducible assembly profiles for steady collocated NS solves."""

from dataclasses import dataclass, replace

from .collocated_pressure_system import (
    CollocatedPressureSystemControls,
    PressureClosureKind,
)
from .collocated_linear_solvers import (
    CollocatedNSLinearSolvers,
    build_collocated_ns_linear_solvers,
)
from .simple_controls import (
    SimpleDiscretizationControls,
    SimpleIterationControls,
)
from .third_party_linear_solver import (
    PetscConstantNullspaceControls,
    ScipyBiCGSTABControls,
)


@dataclass(frozen=True)
class SteadyNSSimpleProfile:
    """Complete construction recipe for one steady SIMPLE route."""

    discretization: SimpleDiscretizationControls
    iteration: SimpleIterationControls
    pressure_system: CollocatedPressureSystemControls
    momentum_linear_solver: ScipyBiCGSTABControls
    pressure_nullspace_linear_solver: (
        PetscConstantNullspaceControls
    )

    def build_linear_solvers(self) -> CollocatedNSLinearSolvers:
        """Build a fresh concrete solver set owned by one model."""
        return build_collocated_ns_linear_solvers(
            momentum_controls=self.momentum_linear_solver,
            pressure_nullspace_controls=(
                self.pressure_nullspace_linear_solver
            ),
        )


def steady_ns_high_accuracy_simple_profile() -> SteadyNSSimpleProfile:
    """Return the production second-order steady SIMPLE profile."""
    return SteadyNSSimpleProfile(
        discretization=SimpleDiscretizationControls(),
        iteration=SimpleIterationControls(),
        pressure_system=CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind.NULLSPACE,
        ),
        momentum_linear_solver=ScipyBiCGSTABControls(
            relative_tolerance=1.0e-9,
            true_residual_tolerance=1.0e-8,
        ),
        pressure_nullspace_linear_solver=(
            PetscConstantNullspaceControls()
        ),
    )


def steady_traction_mms_simple_profile() -> SteadyNSSimpleProfile:
    """Return the named validation profile for the traction MMS."""
    profile = steady_ns_high_accuracy_simple_profile()
    return replace(
        profile,
        iteration=replace(
            profile.iteration,
            max_iterations=1000,
            mass_relative_tolerance=1.0e-8,
        ),
        pressure_system=CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind.GAUGE,
        ),
    )


__all__ = [
    "SteadyNSSimpleProfile",
    "steady_ns_high_accuracy_simple_profile",
    "steady_traction_mms_simple_profile",
]
