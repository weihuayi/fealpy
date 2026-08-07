"""Third-party linear-solver adapters required by FVM baselines."""

from dataclasses import dataclass
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
from fealpy.typing import TensorLike

from .fvm_linear_solver import (
    LinearSolveDiagnostics,
    LinearSolveResult,
)


@dataclass(frozen=True)
class ScipyBiCGSTABControls:
    """Numerical controls for the SciPy BiCGSTAB adapter."""

    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 0.0
    maximum_iterations: int = 1000
    true_residual_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive.")
        if self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be non-negative.")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if self.true_residual_tolerance <= 0.0:
            raise ValueError(
                "true_residual_tolerance must be positive."
            )


@dataclass(frozen=True)
class PetscConstantNullspaceControls:
    """Numerical and cache controls for PETSc constant-nullspace solves."""

    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-14
    maximum_iterations: int = 1000
    true_residual_tolerance: float = 1.0e-7
    cache_matrix_pattern: bool = True

    def __post_init__(self) -> None:
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive.")
        if self.absolute_tolerance <= 0.0:
            raise ValueError("absolute_tolerance must be positive.")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if self.true_residual_tolerance <= 0.0:
            raise ValueError(
                "true_residual_tolerance must be positive."
            )


@dataclass(frozen=True)
class PetscSolveData:
    """Provider data required to validate one PETSc solve."""

    solution: np.ndarray
    convergence_reason: int
    iterations: int
    relative_residual: float


def _validate_system(
    matrix: CSRTensor,
    rhs: TensorLike,
) -> None:
    if not isinstance(matrix, CSRTensor):
        raise TypeError("matrix must be a CSRTensor.")
    if rhs.ndim != 1:
        raise ValueError("rhs must have shape (N,).")
    if matrix.shape[0] != rhs.shape[0]:
        raise ValueError("matrix and rhs sizes must agree.")


class ScipyBiCGSTABSolver:
    """Solve one FVM linear system with SciPy BiCGSTAB."""

    def __init__(
        self,
        controls: ScipyBiCGSTABControls,
    ) -> None:
        self.controls = controls

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        _validate_system(matrix, rhs)
        try:
            from scipy.sparse.linalg import bicgstab
        except ImportError as error:
            raise RuntimeError(
                "scipy bicgstab requires SciPy."
            ) from error

        scipy_matrix = matrix.to_scipy()
        rhs_array = np.asarray(bm.to_numpy(rhs), dtype=float)
        iterations = 0

        def count_iteration(_value=None) -> None:
            nonlocal iterations
            iterations += 1

        solution_array, info = bicgstab(
            scipy_matrix,
            rhs_array,
            rtol=self.controls.relative_tolerance,
            atol=self.controls.absolute_tolerance,
            maxiter=self.controls.maximum_iterations,
            callback=count_iteration,
        )
        info = int(info)
        relative_residual = float(
            np.linalg.norm(scipy_matrix @ solution_array - rhs_array)
            / max(float(np.linalg.norm(rhs_array)), 1.0)
        )
        if info != 0:
            raise RuntimeError(
                "scipy bicgstab failed to converge for matrix "
                f"{scipy_matrix.shape}: info={info}, "
                f"iterations={iterations}, "
                f"true residual={relative_residual:.3e}."
            )
        if (
            relative_residual
            > self.controls.true_residual_tolerance
        ):
            raise RuntimeError(
                "scipy bicgstab failed true residual check for matrix "
                f"{scipy_matrix.shape}: {relative_residual:.3e} > "
                f"{self.controls.true_residual_tolerance:.3e}."
            )
        return LinearSolveResult(
            solution=bm.array(
                solution_array,
                dtype=rhs.dtype,
                device=bm.get_device(rhs),
            ),
            diagnostics=LinearSolveDiagnostics(
                provider="scipy",
                solver="bicgstab",
                iterations=iterations,
                converged=True,
                provider_code=info,
                relative_residual=relative_residual,
            ),
        )

    def close(self) -> None:
        """Close the uniform implementation lifecycle."""
        return None


class PetscConstantNullspaceSolver:
    """Own a serial PETSc constant-nullspace KSP and its pattern cache."""

    def __init__(
        self,
        controls: PetscConstantNullspaceControls,
    ) -> None:
        self.controls = controls
        self._petsc = None
        self._shape = None
        self._indptr = None
        self._indices = None
        self._data = None
        self._matrix = None
        self._nullspace = None
        self._ksp = None

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        _validate_system(matrix, rhs)
        try:
            self._prepare_petsc_matrix(matrix)
            solve_data = self.solve_petsc_system(rhs)

            if solve_data.convergence_reason <= 0:
                raise RuntimeError(
                    "petsc gmres-hypre failed to converge for matrix "
                    f"{matrix.shape}: "
                    f"reason={solve_data.convergence_reason}, "
                    f"iterations={solve_data.iterations}, "
                    "true residual="
                    f"{solve_data.relative_residual:.3e}."
                )
            if (
                solve_data.relative_residual
                > self.controls.true_residual_tolerance
            ):
                raise RuntimeError(
                    "petsc gmres-hypre failed true residual check for "
                    f"matrix {matrix.shape}: "
                    f"{solve_data.relative_residual:.3e} > "
                    f"{self.controls.true_residual_tolerance:.3e}."
                )

            return LinearSolveResult(
                solution=bm.array(
                    solve_data.solution,
                    dtype=rhs.dtype,
                    device=bm.get_device(rhs),
                ),
                diagnostics=LinearSolveDiagnostics(
                    provider="petsc",
                    solver="gmres-hypre",
                    iterations=solve_data.iterations,
                    converged=True,
                    provider_code=solve_data.convergence_reason,
                    relative_residual=(
                        solve_data.relative_residual
                    ),
                ),
            )
        finally:
            if not self.controls.cache_matrix_pattern:
                self._release_petsc_resources()

    def solve_petsc_system(
        self,
        rhs: TensorLike,
    ) -> PetscSolveData:
        """Execute the prepared PETSc KSP and return call-local data."""
        PETSc = self._petsc
        rhs_array = np.asarray(
            bm.to_numpy(rhs),
            dtype=PETSc.ScalarType,
        )
        rhs_vector = None
        solution_vector = None
        residual_vector = None
        try:
            rhs_vector = PETSc.Vec().createWithArray(
                rhs_array,
                comm=PETSc.COMM_SELF,
            )
            self._nullspace.remove(rhs_vector)
            rhs_norm = float(rhs_vector.norm())
            solution_vector = rhs_vector.duplicate()
            solution_vector.set(0.0)
            self._ksp.solve(rhs_vector, solution_vector)
            reason = int(self._ksp.getConvergedReason())
            iterations = int(self._ksp.getIterationNumber())
            solution_array = np.array(
                solution_vector.getArray(readonly=True),
                copy=True,
            )
            residual_vector = rhs_vector.duplicate()
            self._matrix.mult(solution_vector, residual_vector)
            residual_vector.axpy(-1.0, rhs_vector)
            relative_residual = float(
                residual_vector.norm()
                / max(rhs_norm, 1.0)
            )
            return PetscSolveData(
                solution=solution_array,
                convergence_reason=reason,
                iterations=iterations,
                relative_residual=relative_residual,
            )
        finally:
            if residual_vector is not None:
                residual_vector.destroy()
            if solution_vector is not None:
                solution_vector.destroy()
            if rhs_vector is not None:
                rhs_vector.destroy()

    def _prepare_petsc_matrix(self, matrix) -> None:
        if self._matches_cached_pattern(matrix):
            self._update_petsc_matrix_values(matrix)
            return
        self._release_petsc_resources()
        self._build_petsc_resources(matrix)

    def _matches_cached_pattern(self, matrix) -> bool:
        return (
            self._matrix is not None
            and matrix.shape == self._shape
            and np.array_equal(
                bm.to_numpy(matrix.indptr),
                self._indptr,
            )
            and np.array_equal(
                bm.to_numpy(matrix.indices),
                self._indices,
            )
        )

    def _build_petsc_resources(self, matrix) -> None:
        try:
            from petsc4py import PETSc
        except ImportError as error:
            raise RuntimeError(
                "petsc gmres-hypre requires petsc4py."
            ) from error

        self._petsc = PETSc
        try:
            self._shape = matrix.shape
            self._indptr = np.asarray(
                bm.to_numpy(matrix.indptr),
                dtype=PETSc.IntType,
            )
            self._indices = np.asarray(
                bm.to_numpy(matrix.indices),
                dtype=PETSc.IntType,
            )
            self._data = np.asarray(
                bm.to_numpy(matrix.data),
                dtype=PETSc.ScalarType,
            ).copy()
            self._matrix = PETSc.Mat().createAIJ(
                size=matrix.shape,
                csr=(
                    self._indptr,
                    self._indices,
                    self._data,
                ),
                comm=PETSc.COMM_SELF,
            )
            self._matrix.assemble()
            self._nullspace = PETSc.NullSpace().create(
                constant=True,
                comm=PETSc.COMM_SELF,
            )
            self._matrix.setNullSpace(self._nullspace)
            self._matrix.setNearNullSpace(self._nullspace)
            self._ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self._ksp.setOperators(self._matrix)
            self._ksp.setType("gmres")
            self._ksp.getPC().setType("hypre")
            self._ksp.setTolerances(
                rtol=self.controls.relative_tolerance,
                atol=self.controls.absolute_tolerance,
                max_it=self.controls.maximum_iterations,
            )
        except Exception:
            self._release_petsc_resources()
            raise

    def _update_petsc_matrix_values(self, matrix) -> None:
        data = np.asarray(
            bm.to_numpy(matrix.data),
            dtype=self._petsc.ScalarType,
        )
        if np.array_equal(data, self._data):
            return
        self._data = data.copy()
        self._matrix.setValuesCSR(
            self._indptr,
            self._indices,
            self._data,
        )
        self._matrix.assemble()
        self._ksp.setOperators(self._matrix)

    def _release_petsc_resources(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
        if self._nullspace is not None:
            self._nullspace.destroy()
        if self._matrix is not None:
            self._matrix.destroy()
        self._petsc = None
        self._shape = None
        self._indptr = None
        self._indices = None
        self._data = None
        self._matrix = None
        self._nullspace = None
        self._ksp = None

    def close(self) -> None:
        self._release_petsc_resources()


class ThirdPartyLinearSolver:
    """Adapt one explicitly selected third-party solver to FVM tensors."""

    def __init__(
        self,
        implementation: (
            ScipyBiCGSTABSolver
            | PetscConstantNullspaceSolver
        ),
    ) -> None:
        self._implementation = implementation
        self._closed = False

    @classmethod
    def scipy_bicgstab(
        cls,
        controls: ScipyBiCGSTABControls,
    ) -> "ThirdPartyLinearSolver":
        if not isinstance(controls, ScipyBiCGSTABControls):
            raise TypeError(
                "scipy_bicgstab requires ScipyBiCGSTABControls."
            )
        return cls(ScipyBiCGSTABSolver(controls))

    @classmethod
    def petsc_constant_nullspace(
        cls,
        controls: PetscConstantNullspaceControls,
    ) -> "ThirdPartyLinearSolver":
        if not isinstance(
            controls,
            PetscConstantNullspaceControls,
        ):
            raise TypeError(
                "petsc_constant_nullspace requires "
                "PetscConstantNullspaceControls."
            )
        return cls(PetscConstantNullspaceSolver(controls))

    def solve(
        self,
        matrix: CSRTensor,
        rhs: TensorLike,
    ) -> LinearSolveResult:
        if self._closed:
            raise RuntimeError("third-party linear solver is closed.")
        return self._implementation.solve(matrix, rhs)

    def close(self) -> None:
        if self._closed:
            return
        self._implementation.close()
        self._closed = True


__all__ = [
    "PetscConstantNullspaceControls",
    "ScipyBiCGSTABControls",
    "ThirdPartyLinearSolver",
]
