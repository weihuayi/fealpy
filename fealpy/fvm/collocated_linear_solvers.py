"""Concrete linear solvers bound to collocated NS equation roles."""

from dataclasses import dataclass
from typing import TypeAlias

from .fvm_linear_solver import FVMLinearSolver
from .third_party_linear_solver import (
    PetscConstantNullspaceControls,
    ScipyBiCGSTABControls,
    ThirdPartyLinearSolver,
)

LinearSystemSolver: TypeAlias = (
    FVMLinearSolver | ThirdPartyLinearSolver
)


@dataclass(frozen=True)
class CollocatedNSLinearSolvers:
    """Concrete solver objects selected during algorithm construction."""

    momentum: LinearSystemSolver
    pressure_dirichlet: LinearSystemSolver
    pressure_nullspace: LinearSystemSolver
    pressure_gauge: LinearSystemSolver

    def close(self) -> None:
        """Release each owned third-party solver exactly once."""
        closed = set()
        for linear_solver in (
            self.momentum,
            self.pressure_dirichlet,
            self.pressure_nullspace,
            self.pressure_gauge,
        ):
            if (
                isinstance(linear_solver, ThirdPartyLinearSolver)
                and id(linear_solver) not in closed
            ):
                linear_solver.close()
                closed.add(id(linear_solver))


def build_collocated_ns_linear_solvers(
    *,
    momentum_controls: ScipyBiCGSTABControls | None = None,
    pressure_nullspace_controls: (
        PetscConstantNullspaceControls | None
    ) = None,
) -> CollocatedNSLinearSolvers:
    """Build the verified production solver mapping for collocated NS."""
    if momentum_controls is None:
        momentum_controls = ScipyBiCGSTABControls()
    if pressure_nullspace_controls is None:
        pressure_nullspace_controls = (
            PetscConstantNullspaceControls()
        )
    return CollocatedNSLinearSolvers(
        momentum=ThirdPartyLinearSolver.scipy_bicgstab(
            momentum_controls
        ),
        pressure_dirichlet=FVMLinearSolver("scipy"),
        pressure_nullspace=(
            ThirdPartyLinearSolver.petsc_constant_nullspace(
                pressure_nullspace_controls
            )
        ),
        pressure_gauge=FVMLinearSolver("scipy"),
    )


__all__ = [
    "CollocatedNSLinearSolvers",
    "LinearSystemSolver",
    "build_collocated_ns_linear_solvers",
]
