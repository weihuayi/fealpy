from dataclasses import replace

from fealpy.backend import backend_manager as bm
from fealpy.fvm import MassResidualMetrics


def _boundary_convection_rhs(model, uf):
    from fealpy.fvm import FVMGeometry

    geometry = getattr(model, "fvm_geometry", FVMGeometry(model.mesh))
    boundary_faces = bm.nonzero(geometry.is_boundary)[0]
    owner = geometry.owner[boundary_faces]
    face_center = geometry.face_center[boundary_faces]
    velocity_bc = model.pde.dirichlet_velocity(face_center)
    Sf = geometry.S_f[boundary_faces]
    flux = bm.einsum("ij,ij->i", uf[boundary_faces], Sf)

    fx = bm.zeros(model.NC)
    fy = bm.zeros(model.NC)
    bm.add_at(fx, owner, -flux * velocity_bc[:, 0])
    bm.add_at(fy, owner, -flux * velocity_bc[:, 1])
    return bm.concatenate([fx, fy], axis=0)


def test_dirichlet_bc_convection_apply_handles_vector_dirichlet_data():
    bm.set_backend("numpy")
    from fealpy.fvm import DirichletBC
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    face_center = mesh.entity_barycenter("face")
    uf = pde.velocity(face_center)

    class ModelView:
        pass

    model = ModelView()
    model.mesh = mesh
    model.pde = pde
    model.NC = mesh.number_of_cells()
    from fealpy.fvm import FVMGeometry
    model.fvm_geometry = FVMGeometry(mesh)

    b = bm.zeros(2 * model.NC)
    boundary_faces = bm.nonzero(model.fvm_geometry.is_boundary)[0]
    actual = DirichletBC(
        model.fvm_geometry,
        boundary_faces,
        pde.dirichlet_velocity(
            model.fvm_geometry.face_center[boundary_faces]
        ),
        ).apply_convection(b, uf, components=2)
    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(actual - expected))) < 1.0e-12


def test_simple_momentum_rhs_includes_dirichlet_boundary_convection(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_simple_model as simple_model
    from fealpy.fvm import (
        CollocatedNSLinearSolvers,
        LinearSolveDiagnostics,
        LinearSolveResult,
        steady_ns_high_accuracy_simple_profile,
    )

    def first_momentum_rhs(uf):
        captured = []

        class CapturingLinearSolver:
            def solve(self, matrix, rhs):
                captured.append(rhs.copy())
                return LinearSolveResult(
                    solution=bm.zeros(matrix.shape[0]),
                    diagnostics=LinearSolveDiagnostics(
                        provider="test",
                        solver="capture",
                        iterations=None,
                        converged=True,
                        provider_code=None,
                        relative_residual=0.0,
                    ),
                )

        profile = steady_ns_high_accuracy_simple_profile()
        linear_solvers = profile.build_linear_solvers()
        linear_solvers = CollocatedNSLinearSolvers(
            momentum=CapturingLinearSolver(),
            pressure_dirichlet=linear_solvers.pressure_dirichlet,
            pressure_nullspace=linear_solvers.pressure_nullspace,
            pressure_gauge=linear_solvers.pressure_gauge,
        )
        profile = replace(
            profile,
            discretization=replace(
                profile.discretization,
                face_flux_correction_scheme="none",
                face_flux_quadrature_order=3,
            ),
            iteration=replace(
                profile.iteration,
                momentum_nonorthogonal_max_iterations=0,
            ),
        )

        model = simple_model.NSFVMSimpleModel(
            {
                "pde": 6,
                "nx": 4,
                "ny": 4,
                "log_level": "ERROR",
                "pbar_log": False,
                "profile": profile,
                "linear_solvers": linear_solvers,
            }
        )
        model.solver.momentum.predict(
            bm.zeros(model.NC),
            uf,
            bm.zeros((model.NC, model.GD)),
            pressure_gradient=bm.zeros((model.NC, model.GD)),
            nonorthogonal_tolerance=(
                model.solver.iteration_controls.momentum_nonorthogonal_rtol
            ),
        )
        return model, bm.concatenate(captured, axis=0)

    zero_model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 4,
            "ny": 4,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    face_center = zero_model.fvm_geometry.face_center
    uf = zero_model.pde.velocity(face_center)
    zero_uf = bm.zeros_like(uf)

    _, rhs_zero = first_momentum_rhs(zero_uf)
    model, rhs_with_boundary_flux = first_momentum_rhs(uf)

    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(rhs_with_boundary_flux - rhs_zero - expected))) < 1.0e-12


def test_simple_mass_residual_is_normalized_and_scale_invariant():
    bm.set_backend("numpy")
    from fealpy.fvm import FVMGeometry, collocated_mass_residual
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=20, ny=20)
    geometry = FVMGeometry(mesh)
    face_velocity = pde.velocity(mesh.entity_barycenter("face"))

    residual = collocated_mass_residual(
        face_velocity,
        geometry=geometry,
    )
    scaled_residual = collocated_mass_residual(
        7.0 * face_velocity,
        geometry=geometry,
    )

    assert residual < 1.0e-3
    assert abs(residual - scaled_residual) < 1.0e-12


def test_collocated_simple_records_common_residuals(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.ns_fvm_simple_model as simple_model
    from fealpy.fvm.collocated_momentum_equation import (
        MomentumBalance,
        MomentumPredictorResult,
    )
    from fealpy.fvm.collocated_pressure_system import (
        PressureCorrectionResult,
    )

    class FakeRhieChow:
        def __init__(self, geometry, boundary, **kwargs):
            self.mesh = geometry.mesh

        def apply(
            self,
            base_face_velocity,
            pressure,
            face_response,
            pressure_gradient,
        ):
            return base_face_velocity

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(
        simple_solver,
        "collocated_mass_metrics",
        lambda uf, **kwargs: MassResidualMetrics(
            relative_l1=0.0,
            relative_l2=0.0,
            divergence_l2=0.0,
            absolute_linf=0.0,
        ),
    )
    monkeypatch.setattr(
        simple_solver,
        "cell_l2_norm",
        lambda value, **kwargs: 0.0,
    )
    from fealpy.fvm import steady_ns_high_accuracy_simple_profile

    profile = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        profile,
        iteration=replace(profile.iteration, max_iterations=5),
    )
    model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "log_level": "ERROR",
            "pbar_log": False,
            "profile": profile,
        }
    )
    model.pde.dirichlet_velocity = lambda points: bm.zeros_like(points)
    monkeypatch.setattr(
        model.solver.momentum,
        "predict",
        lambda p, uf, u0, **kwargs: MomentumPredictorResult(
            cell_velocity=bm.zeros((model.NC, model.GD)),
            correction_diagonal=bm.ones(model.NC),
            spatial_diagonal=bm.ones(model.NC),
            nonorthogonal_iterations=0,
            nonorthogonal_tolerance=1.0e-4,
            nonorthogonal_residual=None,
            linear_solves=(),
        ),
    )
    monkeypatch.setattr(
        model.solver.pressure_system,
        "solve",
        lambda uf, ap: PressureCorrectionResult(
            pressure_correction=bm.zeros(model.NC),
            nonorthogonal_iterations=0,
            nonorthogonal_residual=None,
            nonorthogonal_relative_update=0.0,
            linear_solves=(),
        ),
    )
    monkeypatch.setattr(
        model.solver.momentum,
        "balance",
        lambda p, u, uf, *, pressure_gradient: MomentumBalance(
            lhs=bm.zeros(model.GD * model.NC),
            rhs=bm.zeros(model.GD * model.NC),
            residual=bm.zeros(model.GD * model.NC),
        ),
    )

    result = model.solve()

    assert len(result.residual_history) == 1
    assert result.residual_history[0].mass_relative_l2 == 0.0
    assert result.residual_history[0].pressure_correction_l2 == 0.0


def test_collocated_simple_updates_cell_velocity_after_pressure_correction(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.ns_fvm_simple_model as simple_model
    from fealpy.fvm.collocated_momentum_equation import (
        MomentumBalance,
        MomentumPredictorResult,
    )
    from fealpy.fvm.collocated_pressure_system import (
        PressureCorrectionResult,
    )

    class FakeRhieChow:
        def __init__(self, geometry, boundary, **kwargs):
            self.mesh = geometry.mesh

        def apply(
            self,
            base_face_velocity,
            pressure,
            face_response,
            pressure_gradient,
        ):
            return base_face_velocity

    calls = []

    def fake_temporary_velocity(p, uf, u0, **kwargs):
        calls.append(u0.copy())
        response = bm.ones(model.NC)
        return MomentumPredictorResult(
            cell_velocity=bm.zeros((model.NC, model.GD)),
            correction_diagonal=response,
            spatial_diagonal=response,
            nonorthogonal_iterations=0,
            nonorthogonal_tolerance=1.0e-4,
            nonorthogonal_residual=None,
            linear_solves=(),
        )

    def fake_velocity_pressure_correction(
        cell_velocity,
        pressure_gradient,
        cell_response,
    ):
        assert pressure_gradient.shape == (model.NC, model.GD)
        assert cell_response.shape == (model.NC,)
        return cell_velocity + 3.0

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(
        simple_solver,
        "collocated_mass_metrics",
        lambda uf, **kwargs: MassResidualMetrics(
            relative_l1=0.0,
            relative_l2=0.0,
            divergence_l2=0.0,
            absolute_linf=0.0,
        ),
    )
    monkeypatch.setattr(
        simple_solver,
        "cell_l2_norm",
        lambda value, **kwargs: 1.0,
    )
    from fealpy.fvm import steady_ns_high_accuracy_simple_profile

    profile = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        profile,
        iteration=replace(
            profile.iteration,
            max_iterations=1,
            pressure_relaxation=0.5,
            momentum_relative_tolerance=1.0e-12,
            mass_relative_tolerance=1.0e-99,
        ),
    )
    model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "log_level": "ERROR",
            "pbar_log": False,
            "profile": profile,
        }
    )
    model.pde.dirichlet_velocity = lambda points: bm.zeros_like(points)
    monkeypatch.setattr(
        simple_solver,
        "correct_cell_velocity",
        fake_velocity_pressure_correction,
    )
    monkeypatch.setattr(model.solver.momentum, "predict", fake_temporary_velocity)
    monkeypatch.setattr(
        model.solver.pressure_system,
        "solve",
        lambda uf, ap: PressureCorrectionResult(
            pressure_correction=bm.ones(model.NC),
            nonorthogonal_iterations=0,
            nonorthogonal_residual=None,
            nonorthogonal_relative_update=0.0,
            linear_solves=(),
        ),
    )
    monkeypatch.setattr(
        model.solver.momentum,
        "balance",
        lambda p, u, uf, *, pressure_gradient: MomentumBalance(
            lhs=bm.ones(model.GD * model.NC),
            rhs=bm.zeros(model.GD * model.NC),
            residual=bm.ones(model.GD * model.NC),
        ),
    )

    result = model.solve()

    assert len(calls) == 1
    assert bm.max(bm.abs(result.velocity - 3.0)) < 1.0e-14


def test_collocated_simple_avoids_duplicate_face_pressure_correction(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.ns_fvm_simple_model as simple_model
    from fealpy.fvm.collocated_momentum_equation import (
        MomentumBalance,
        MomentumPredictorResult,
    )
    from fealpy.fvm.collocated_pressure_system import (
        PressureCorrectionResult,
    )

    class FakeRhieChow:
        def __init__(self, geometry, boundary, **kwargs):
            self.mesh = geometry.mesh

        def apply(
            self,
            base_face_velocity,
            pressure,
            face_response,
            pressure_gradient,
        ):
            return base_face_velocity

    def fake_temporary_velocity(p, uf, u0, **kwargs):
        response = bm.ones(model.NC)
        return MomentumPredictorResult(
            cell_velocity=bm.zeros((model.NC, model.GD)),
            correction_diagonal=response,
            spatial_diagonal=response,
            nonorthogonal_iterations=0,
            nonorthogonal_tolerance=1.0e-4,
            nonorthogonal_residual=None,
            linear_solves=(),
        )

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(
        simple_solver,
        "collocated_mass_metrics",
        lambda uf, **kwargs: MassResidualMetrics(
            relative_l1=0.0,
            relative_l2=0.0,
            divergence_l2=0.0,
            absolute_linf=0.0,
        ),
    )
    monkeypatch.setattr(
        simple_solver,
        "cell_l2_norm",
        lambda value, **kwargs: 1.0,
    )

    from fealpy.fvm import steady_ns_high_accuracy_simple_profile

    profile = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        profile,
        iteration=replace(
            profile.iteration,
            max_iterations=1,
            pressure_relaxation=0.5,
            momentum_relative_tolerance=1.0e-12,
            mass_relative_tolerance=1.0e-99,
        ),
    )
    model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "log_level": "ERROR",
            "pbar_log": False,
            "profile": profile,
        }
    )
    model.pde.dirichlet_velocity = lambda points: bm.zeros_like(points)
    monkeypatch.setattr(model.solver.momentum, "predict", fake_temporary_velocity)
    monkeypatch.setattr(
        model.solver.pressure_system,
        "solve",
        lambda uf, ap: PressureCorrectionResult(
            pressure_correction=bm.ones(model.NC),
            nonorthogonal_iterations=0,
            nonorthogonal_residual=None,
            nonorthogonal_relative_update=0.0,
            linear_solves=(),
        ),
    )
    monkeypatch.setattr(
        model.solver.momentum,
        "balance",
        lambda p, u, uf, *, pressure_gradient: MomentumBalance(
            lhs=bm.zeros(model.GD * model.NC),
            rhs=bm.zeros(model.GD * model.NC),
            residual=bm.zeros(model.GD * model.NC),
        ),
    )

    model.solve()
