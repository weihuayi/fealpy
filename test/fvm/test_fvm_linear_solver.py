from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor


def diagonal_matrix(values) -> CSRTensor:
    values = bm.array(values, dtype=bm.float64)
    size = values.shape[0]
    return CSRTensor(
        bm.arange(size + 1, dtype=bm.int64),
        bm.arange(size, dtype=bm.int64),
        values,
        spshape=(size, size),
    )


def test_fvm_linear_solver_returns_call_local_result():
    from fealpy.fvm.fvm_linear_solver import (
        FVMLinearSolver,
        LinearSolveDiagnostics,
        LinearSolveResult,
    )

    solver = FVMLinearSolver("scipy")
    result = solver.solve(
        diagonal_matrix([2.0, 4.0]),
        bm.array([4.0, 8.0], dtype=bm.float64),
    )

    assert isinstance(result, LinearSolveResult)
    assert isinstance(result.diagnostics, LinearSolveDiagnostics)
    np.testing.assert_allclose(
        bm.to_numpy(result.solution),
        [2.0, 2.0],
    )
    assert result.diagnostics.provider == "fealpy"
    assert result.diagnostics.solver == "scipy-direct"
    assert result.diagnostics.iterations is None
    assert result.diagnostics.converged
    assert result.diagnostics.provider_code is None
    assert result.diagnostics.relative_residual is None


def test_linear_solve_result_is_immutable():
    from fealpy.fvm.fvm_linear_solver import (
        LinearSolveDiagnostics,
        LinearSolveResult,
    )

    result = LinearSolveResult(
        solution=bm.array([1.0]),
        diagnostics=LinearSolveDiagnostics(
            provider="fealpy",
            solver="scipy-direct",
            iterations=None,
            converged=True,
            provider_code=None,
            relative_residual=None,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        result.diagnostics = result.diagnostics


def test_fvm_linear_solver_rejects_unverified_or_automatic_routes():
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver

    for route in ("auto", "mumps", "cupy", "scipy_bicgstab"):
        with pytest.raises(ValueError, match="verified FEALPy solver"):
            FVMLinearSolver(route)


def test_fvm_linear_solver_reports_provider_and_matrix_on_failure(
    monkeypatch,
):
    import fealpy.fvm.fvm_linear_solver as module

    def failed_solve(*args, **kwargs):
        raise ValueError("provider failure")

    monkeypatch.setattr(module, "spsolve", failed_solve)
    with pytest.raises(
        RuntimeError,
        match=r"fealpy scipy-direct failed for matrix \(2, 2\)",
    ):
        module.FVMLinearSolver("scipy").solve(
            diagonal_matrix([2.0, 4.0]),
            bm.array([4.0, 8.0]),
        )


def test_fvm_linear_solver_isolates_matrix_from_provider_mutation(
    monkeypatch,
):
    import fealpy.fvm.fvm_linear_solver as module

    matrix = diagonal_matrix([2.0, 4.0])
    original_values = bm.copy(matrix.values)

    def mutating_solve(candidate, rhs, **kwargs):
        candidate._values[0] = -999.0
        return bm.array([2.0, 2.0], dtype=rhs.dtype)

    monkeypatch.setattr(module, "spsolve", mutating_solve)
    result = module.FVMLinearSolver("scipy").solve(
        matrix,
        bm.array([4.0, 8.0], dtype=bm.float64),
    )

    np.testing.assert_allclose(
        bm.to_numpy(result.solution),
        [2.0, 2.0],
    )
    np.testing.assert_array_equal(
        bm.to_numpy(matrix.values),
        bm.to_numpy(original_values),
    )


def test_named_constructors_bind_concrete_solver_classes():
    import fealpy.fvm.third_party_linear_solver as module

    scipy_solver = module.ThirdPartyLinearSolver.scipy_bicgstab(
        module.ScipyBiCGSTABControls()
    )
    petsc_solver = (
        module.ThirdPartyLinearSolver.petsc_constant_nullspace(
            module.PetscConstantNullspaceControls()
        )
    )

    assert isinstance(
        scipy_solver._implementation,
        module.ScipyBiCGSTABSolver,
    )
    assert isinstance(
        petsc_solver._implementation,
        module.PetscConstantNullspaceSolver,
    )
    scipy_solver.close()
    petsc_solver.close()


def test_scipy_bicgstab_returns_true_residual_diagnostics():
    from fealpy.fvm.third_party_linear_solver import (
        ScipyBiCGSTABControls,
        ThirdPartyLinearSolver,
    )

    solver = ThirdPartyLinearSolver.scipy_bicgstab(
        ScipyBiCGSTABControls(
            relative_tolerance=1.0e-12,
            true_residual_tolerance=1.0e-11,
        )
    )
    rhs = bm.array([4.0, 8.0], dtype=bm.float64)
    result = solver.solve(diagonal_matrix([2.0, 4.0]), rhs)

    np.testing.assert_allclose(
        bm.to_numpy(result.solution),
        [2.0, 2.0],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert result.solution.dtype == rhs.dtype
    assert bm.get_device(result.solution) == bm.get_device(rhs)
    assert result.diagnostics.provider == "scipy"
    assert result.diagnostics.solver == "bicgstab"
    assert result.diagnostics.iterations >= 0
    assert result.diagnostics.converged
    assert result.diagnostics.provider_code == 0
    assert result.diagnostics.relative_residual <= 1.0e-11


def test_scipy_solver_uses_csr_returned_by_fealpy(
    monkeypatch,
):
    import scipy.sparse.linalg

    from fealpy.fvm.third_party_linear_solver import (
        ScipyBiCGSTABControls,
        ScipyBiCGSTABSolver,
    )

    class ReturnedScipyMatrix:
        shape = (2, 2)

        def tocsr(self):
            raise AssertionError("to_scipy already returned CSR")

        def __matmul__(self, vector):
            return np.array(
                [2.0 * vector[0], 4.0 * vector[1]]
            )

    returned_matrix = ReturnedScipyMatrix()
    matrix = diagonal_matrix([2.0, 4.0])
    monkeypatch.setattr(
        matrix,
        "to_scipy",
        lambda: returned_matrix,
    )
    calls = []

    def solve_bicgstab(candidate, rhs, **kwargs):
        calls.append(candidate)
        return np.array([2.0, 2.0]), 0

    monkeypatch.setattr(
        scipy.sparse.linalg,
        "bicgstab",
        solve_bicgstab,
    )
    result = ScipyBiCGSTABSolver(
        ScipyBiCGSTABControls()
    ).solve(
        matrix,
        bm.array([4.0, 8.0], dtype=bm.float64),
    )

    assert calls == [returned_matrix]
    np.testing.assert_allclose(result.solution, [2.0, 2.0])


def test_scipy_bicgstab_failure_does_not_fallback(monkeypatch):
    import scipy.sparse.linalg

    from fealpy.fvm.third_party_linear_solver import (
        ScipyBiCGSTABControls,
        ThirdPartyLinearSolver,
    )

    def failed_bicgstab(*args, **kwargs):
        return np.zeros(2), 7

    monkeypatch.setattr(
        scipy.sparse.linalg,
        "bicgstab",
        failed_bicgstab,
    )
    solver = ThirdPartyLinearSolver.scipy_bicgstab(
        ScipyBiCGSTABControls()
    )

    with pytest.raises(RuntimeError, match="info=7"):
        solver.solve(
            diagonal_matrix([2.0, 4.0]),
            bm.array([4.0, 8.0], dtype=bm.float64),
        )


def test_petsc_constant_nullspace_solver_owns_reusable_resource(monkeypatch):
    import fealpy.fvm.third_party_linear_solver as module
    from fealpy.fvm.fvm_linear_solver import (
        LinearSolveDiagnostics,
        LinearSolveResult,
    )
    from fealpy.fvm.third_party_linear_solver import (
        PetscConstantNullspaceControls,
        ThirdPartyLinearSolver,
    )

    calls = []

    class FakePetscSolver:
        def __init__(self, controls):
            calls.append(("init", controls))

        def solve(self, matrix, rhs):
            calls.append(("solve", matrix, tuple(bm.to_numpy(rhs))))
            return LinearSolveResult(
                solution=rhs,
                diagnostics=LinearSolveDiagnostics(
                    provider="petsc",
                    solver="gmres-hypre",
                    iterations=3,
                    converged=True,
                    provider_code=2,
                    relative_residual=1.0e-12,
                ),
            )

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        module,
        "PetscConstantNullspaceSolver",
        FakePetscSolver,
    )
    controls = PetscConstantNullspaceControls()
    solver = ThirdPartyLinearSolver.petsc_constant_nullspace(controls)
    rhs = bm.array([1.0, -1.0])

    first = solver.solve("A", rhs)
    second = solver.solve("B", rhs)
    solver.close()
    solver.close()

    assert first.diagnostics.iterations == 3
    assert second.diagnostics.iterations == 3
    assert calls == [
        ("init", controls),
        ("solve", "A", (1.0, -1.0)),
        ("solve", "B", (1.0, -1.0)),
        ("close",),
    ]
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve("C", rhs)


def test_petsc_prepare_matrix_updates_or_rebuilds_one_resource(
    monkeypatch,
):
    from fealpy.fvm.third_party_linear_solver import (
        PetscConstantNullspaceControls,
        PetscConstantNullspaceSolver,
    )

    solver = PetscConstantNullspaceSolver(
        PetscConstantNullspaceControls()
    )
    calls = []
    matrix = object()

    monkeypatch.setattr(
        solver,
        "_matches_cached_pattern",
        lambda candidate: True,
    )
    monkeypatch.setattr(
        solver,
        "_update_petsc_matrix_values",
        lambda candidate: calls.append(("update", candidate)),
    )
    monkeypatch.setattr(
        solver,
        "_release_petsc_resources",
        lambda: calls.append(("destroy",)),
    )
    monkeypatch.setattr(
        solver,
        "_build_petsc_resources",
        lambda candidate: calls.append(("build", candidate)),
    )

    solver._prepare_petsc_matrix(matrix)
    assert calls == [("update", matrix)]

    calls.clear()
    monkeypatch.setattr(
        solver,
        "_matches_cached_pattern",
        lambda candidate: False,
    )
    solver._prepare_petsc_matrix(matrix)
    assert calls == [("destroy",), ("build", matrix)]


def test_petsc_no_cache_releases_resources_when_solve_fails(
    monkeypatch,
):
    from fealpy.fvm.third_party_linear_solver import (
        PetscConstantNullspaceControls,
        PetscConstantNullspaceSolver,
    )

    solver = PetscConstantNullspaceSolver(
        PetscConstantNullspaceControls(
            cache_matrix_pattern=False,
        )
    )
    calls = []
    monkeypatch.setattr(
        solver,
        "_prepare_petsc_matrix",
        lambda matrix: calls.append(("prepare", matrix.shape)),
    )

    def fail_solve(rhs):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(solver, "solve_petsc_system", fail_solve)
    monkeypatch.setattr(
        solver,
        "_release_petsc_resources",
        lambda: calls.append(("destroy",)),
    )

    with pytest.raises(RuntimeError, match="provider failure"):
        solver.solve(
            diagonal_matrix([2.0, 4.0]),
            bm.array([4.0, 8.0], dtype=bm.float64),
        )

    assert calls == [("prepare", (2, 2)), ("destroy",)]


def test_petsc_solver_does_not_construct_scipy_matrix(
    monkeypatch,
):
    from types import SimpleNamespace

    from fealpy.fvm.third_party_linear_solver import (
        PetscConstantNullspaceControls,
        PetscConstantNullspaceSolver,
    )

    matrix = diagonal_matrix([2.0, 4.0])

    def reject_scipy_conversion():
        raise AssertionError("PETSc must consume FEALPy CSR arrays")

    monkeypatch.setattr(
        matrix,
        "to_scipy",
        reject_scipy_conversion,
    )
    solver = PetscConstantNullspaceSolver(
        PetscConstantNullspaceControls()
    )
    calls = []
    monkeypatch.setattr(
        solver,
        "_prepare_petsc_matrix",
        lambda candidate: calls.append(candidate),
    )
    monkeypatch.setattr(
        solver,
        "solve_petsc_system",
        lambda rhs: SimpleNamespace(
            solution=np.array([2.0, 2.0]),
            convergence_reason=3,
            iterations=1,
            relative_residual=0.0,
        ),
    )

    result = solver.solve(
        matrix,
        bm.array([4.0, 8.0], dtype=bm.float64),
    )

    assert calls == [matrix]
    np.testing.assert_allclose(result.solution, [2.0, 2.0])
    assert result.diagnostics.relative_residual == 0.0


def test_petsc_system_computes_true_residual_with_petsc_mat():
    from fealpy.fvm.third_party_linear_solver import (
        PetscConstantNullspaceControls,
        PetscConstantNullspaceSolver,
    )

    destroyed = []

    class FakeVector:
        def __init__(self, values):
            self.values = np.array(values, dtype=float, copy=True)

        def duplicate(self):
            return FakeVector(np.zeros_like(self.values))

        def set(self, value):
            self.values.fill(value)

        def getArray(self, readonly=True):
            return self.values

        def norm(self):
            return float(np.linalg.norm(self.values))

        def axpy(self, coefficient, other):
            self.values += coefficient * other.values

        def destroy(self):
            destroyed.append(self)

    class FakeVectorFactory:
        def createWithArray(self, values, comm):
            return FakeVector(values)

    class FakePetsc:
        ScalarType = np.float64
        COMM_SELF = object()
        Vec = FakeVectorFactory

    class FakeNullspace:
        def remove(self, vector):
            return None

    class FakeKSP:
        def solve(self, rhs, solution):
            solution.values[:] = [2.0, 2.0]

        def getConvergedReason(self):
            return 3

        def getIterationNumber(self):
            return 1

    class FakeMatrix:
        def mult(self, solution, output):
            output.values[:] = [
                2.0 * solution.values[0],
                4.0 * solution.values[1],
            ]

    solver = PetscConstantNullspaceSolver(
        PetscConstantNullspaceControls()
    )
    solver._petsc = FakePetsc
    solver._nullspace = FakeNullspace()
    solver._ksp = FakeKSP()
    solver._matrix = FakeMatrix()

    solve_data = solver.solve_petsc_system(
        bm.array([4.0, 8.0], dtype=bm.float64)
    )

    np.testing.assert_allclose(solve_data.solution, [2.0, 2.0])
    assert solve_data.convergence_reason == 3
    assert solve_data.iterations == 1
    assert solve_data.relative_residual == 0.0
    assert len(destroyed) == 3


def test_petsc_partial_build_failure_releases_created_matrix(
    monkeypatch,
):
    import sys
    from types import SimpleNamespace

    from fealpy.fvm.third_party_linear_solver import (
        PetscConstantNullspaceControls,
        PetscConstantNullspaceSolver,
    )

    calls = []

    class FakeMatrix:
        def createAIJ(self, **kwargs):
            return self

        def assemble(self):
            calls.append(("assemble",))

        def destroy(self):
            calls.append(("destroy",))

    class FakeNullSpace:
        def create(self, **kwargs):
            raise RuntimeError("nullspace construction failed")

    fake_petsc = SimpleNamespace(
        IntType=np.int64,
        ScalarType=np.float64,
        COMM_SELF=object(),
        Mat=FakeMatrix,
        NullSpace=FakeNullSpace,
    )
    monkeypatch.setitem(
        sys.modules,
        "petsc4py",
        SimpleNamespace(PETSc=fake_petsc),
    )
    solver = PetscConstantNullspaceSolver(
        PetscConstantNullspaceControls()
    )

    with pytest.raises(
        RuntimeError,
        match="nullspace construction failed",
    ):
        solver._build_petsc_resources(
            diagonal_matrix([2.0, 4.0])
        )

    assert calls == [("assemble",), ("destroy",)]
    assert solver._matrix is None
