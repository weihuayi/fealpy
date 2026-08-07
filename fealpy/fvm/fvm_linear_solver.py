"""FEALPy linear-solver adapter for finite-volume systems."""

from dataclasses import dataclass
from typing import Literal

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike


@dataclass(frozen=True)
class LinearSolveDiagnostics:
    """Diagnostics produced by one linear-system solve."""

    provider: str
    solver: str
    iterations: int | None
    converged: bool
    provider_code: int | None
    relative_residual: float | None


@dataclass(frozen=True)
class LinearSolveResult:
    """Solution and diagnostics sharing one call-local lifetime."""

    solution: TensorLike
    diagnostics: LinearSolveDiagnostics


class FVMLinearSolver:
    """Adapt an FVM sparse system to a verified ``fealpy.solver`` route."""

    def __init__(
        self,
        solver: Literal["scipy"] = "scipy",
    ) -> None:
        if solver != "scipy":
            raise ValueError(
                "verified FEALPy solver must currently be 'scipy'."
            )
        self.solver = solver

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        """Solve one sparse system through ``fealpy.solver.spsolve``."""
        if not isinstance(matrix, CSRTensor):
            raise TypeError("matrix must be a CSRTensor.")
        if rhs.ndim != 1:
            raise ValueError("rhs must have shape (N,).")
        if matrix.shape[0] != rhs.shape[0]:
            raise ValueError("matrix and rhs sizes must agree.")

        try:
            solution = spsolve(
                matrix.copy(),
                rhs,
                solver=self.solver,
            )
        except Exception as error:
            raise RuntimeError(
                "fealpy scipy-direct failed for matrix "
                f"{matrix.shape}."
            ) from error
        solution = bm.array(
            solution,
            dtype=rhs.dtype,
            device=bm.get_device(rhs),
        )
        return LinearSolveResult(
            solution=solution,
            diagnostics=LinearSolveDiagnostics(
                provider="fealpy",
                solver=f"{self.solver}-direct",
                iterations=None,
                converged=True,
                provider_code=None,
                relative_residual=None,
            ),
        )


__all__ = [
    "FVMLinearSolver",
    "LinearSolveDiagnostics",
    "LinearSolveResult",
]
