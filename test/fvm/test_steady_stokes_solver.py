import numpy as np
from dataclasses import replace

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.fvm import (
    CollocatedPressureSystemControls,
    CollocatedSimpleSolver,
    PDEBoundaryConditions,
    PressureClosureKind,
    SimpleDiscretizationControls,
    SimpleIterationControls,
    StokesFVMSimpleModel,
    build_collocated_ns_linear_solvers,
    resolve_simple_boundary_conditions,
    steady_ns_high_accuracy_simple_profile,
)
from fealpy.model.navier_stokes.exp0012 import Exp0012


class ThreeDimensionalStokesMMS(Exp0012):
    @cartesian
    def source(self, points):
        return -self.viscosity() * self.lap_velocity(points) + self.grad_pressure(points)


def _solve_stokes_model(*, momentum_relaxation=0.7, pressure_relaxation=0.3):
    base = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        base,
        iteration=replace(
            base.iteration,
            max_iterations=400,
            pressure_relaxation=pressure_relaxation,
            momentum_equation_relaxation=momentum_relaxation,
            momentum_relative_tolerance=1.0e-8,
            mass_relative_tolerance=1.0e-8,
        ),
        pressure_system=CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind.GAUGE,
        ),
    )
    model = StokesFVMSimpleModel(
        {
            "pde": 1,
            "nx": 2,
            "ny": 2,
            "profile": profile,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    return model, model.solve()


def test_stokes_simple_reaches_momentum_and_mass_fixed_point():
    bm.set_backend("numpy")
    model, result = _solve_stokes_model()
    residual = result.residual_history[-1]
    errors = model.compute_error(result)

    assert result.converged is True
    assert result.termination_reason == "fixed_point_residuals"
    assert residual.mass_relative_l2 < 1.0e-7
    assert residual.momentum_relative_l2 < 1.0e-7
    assert result.velocity.shape == (model.NC, model.GD)
    assert result.pressure.shape == (model.NC,)
    assert all(np.isfinite(float(error)) for error in errors)
    assert max(float(error) for error in errors) < 1.0


def test_stokes_fixed_point_is_independent_of_relaxation_parameters():
    bm.set_backend("numpy")
    first, first_result = _solve_stokes_model(
        momentum_relaxation=0.5,
        pressure_relaxation=0.2,
    )
    second, second_result = _solve_stokes_model(
        momentum_relaxation=0.8,
        pressure_relaxation=0.4,
    )
    pressure_first = first_result.pressure - bm.mean(
        first_result.pressure
    )
    pressure_second = second_result.pressure - bm.mean(
        second_result.pressure
    )

    assert first_result.converged is True
    assert second_result.converged is True
    np.testing.assert_allclose(
        bm.to_numpy(first_result.velocity),
        bm.to_numpy(second_result.velocity),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        bm.to_numpy(pressure_first),
        bm.to_numpy(pressure_second),
        rtol=0.0,
        atol=1.0e-5,
    )


def test_collocated_stokes_runs_on_three_dimensional_tetrahedra():
    from fealpy.model.navier_stokes.exp0012 import Exp0012

    bm.set_backend("numpy")
    pde = Exp0012({"mu": 1.0})
    mesh = pde.init_mesh["uniform_tet"](nx=2, ny=2, nz=2)
    viscosity = float(bm.to_numpy(pde.viscosity()))

    @cartesian
    def source(points):
        return -viscosity * pde.lap_velocity(points) + pde.grad_pressure(points)

    discretization = SimpleDiscretizationControls()
    iteration = SimpleIterationControls(
        max_iterations=1,
        pressure_relaxation=0.3,
        momentum_relative_tolerance=1.0e-6,
        mass_relative_tolerance=1.0e-6,
    )
    pressure_system = CollocatedPressureSystemControls(
        pure_neumann_closure=PressureClosureKind.GAUGE,
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=pde.dirichlet_velocity,
    )
    solver = CollocatedSimpleSolver(
        diffusion_coef=viscosity,
        convection_coef=0.0,
        source=source,
        boundary_conditions=resolve_simple_boundary_conditions(
            mesh,
            boundary,
            discretization,
            pressure_system,
        ),
        discretization_controls=discretization,
        iteration_controls=iteration,
        linear_solvers=build_collocated_ns_linear_solvers(),
    )

    result = solver.solve()

    assert result.velocity.shape == (mesh.number_of_cells(), 3)
    assert result.face_velocity.shape == (mesh.number_of_faces(), 3)
    assert result.face_flux.shape == (mesh.number_of_faces(),)
    assert result.pressure.shape == (mesh.number_of_cells(),)
    assert bool(bm.all(bm.isfinite(result.velocity)))
    assert bool(bm.all(bm.isfinite(result.pressure)))


def test_stokes_model_forwards_three_dimensional_mesh_parameters():
    bm.set_backend("numpy")
    base = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        base,
        pressure_system=CollocatedPressureSystemControls(
            pure_neumann_closure=PressureClosureKind.GAUGE,
        ),
    )
    model = StokesFVMSimpleModel(
        {
            "pde": ThreeDimensionalStokesMMS({"mu": 1.0}),
            "mesh_type": "uniform_tet",
            "nx": 2,
            "ny": 2,
            "nz": 3,
            "profile": profile,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    assert model.GD == 3
    assert model.NC == 72
