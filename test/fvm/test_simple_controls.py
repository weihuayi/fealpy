from dataclasses import FrozenInstanceError, asdict

import pytest


def test_simple_discretization_controls_freeze_high_accuracy_defaults():
    from fealpy.fvm.simple_controls import SimpleDiscretizationControls

    controls = SimpleDiscretizationControls()

    assert asdict(controls) == {
        "pressure_gradient_method": "layered_lsq",
        "velocity_gradient_method": "layered_lsq",
        "gradient_layer_weights": (1.0, 0.25),
        "gradient_boundary_weight": 1.0,
        "diffusion_method": "over_relaxed",
        "diffusion_nonorthogonal_eps": 0.05,
        "momentum_face_interpolation": "average",
        "pressure_response_interpolation": "average",
        "rhie_chow_velocity_interpolation": "average",
        "spatial_face_velocity_scheme": "second_order_reconstructed",
        "face_flux_correction_scheme": "cell_anchored_quadratic",
        "face_flux_quadrature_order": 5,
        "face_flux_max_stencil_layers": 4,
        "face_flux_max_condition": 100.0,
    }
    with pytest.raises(FrozenInstanceError):
        controls.diffusion_method = "orthogonal"


def test_simple_iteration_controls_freeze_high_accuracy_defaults():
    from fealpy.fvm.simple_controls import SimpleIterationControls

    assert asdict(SimpleIterationControls()) == {
        "max_iterations": 1500,
        "pressure_relaxation": 0.3,
        "momentum_equation_relaxation": 0.9,
        "momentum_relative_tolerance": 1.0e-7,
        "mass_relative_tolerance": 1.0e-7,
        "momentum_nonorthogonal_max_iterations": 50,
        "momentum_nonorthogonal_rtol": 1.0e-4,
        "momentum_nonorthogonal_atol": 1.0e-12,
        "pressure_nonorthogonal_max_iterations": 50,
        "pressure_nonorthogonal_rtol": 1.0e-5,
        "pressure_nonorthogonal_atol": 1.0e-12,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"momentum_face_interpolation": None}, "momentum_face_interpolation"),
        ({"pressure_response_interpolation": "cubic"}, "pressure_response_interpolation"),
        ({"face_flux_quadrature_order": 0}, "face_flux_quadrature_order"),
    ],
)
def test_simple_discretization_controls_reject_noncanonical_values(kwargs, message):
    from fealpy.fvm.simple_controls import SimpleDiscretizationControls

    with pytest.raises((TypeError, ValueError), match=message):
        SimpleDiscretizationControls(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"pressure_relaxation": 0.0}, "pressure_relaxation"),
        ({"momentum_relative_tolerance": 0.0}, "momentum_relative_tolerance"),
        (
            {"pressure_nonorthogonal_max_iterations": -1},
            "pressure_nonorthogonal_max_iterations",
        ),
    ],
)
def test_simple_iteration_controls_reject_invalid_values(kwargs, message):
    from fealpy.fvm.simple_controls import SimpleIterationControls

    with pytest.raises(ValueError, match=message):
        SimpleIterationControls(**kwargs)


def test_collocated_pressure_and_linear_solver_controls_are_concrete():
    from fealpy.fvm import (
        FVMLinearSolver,
        ThirdPartyLinearSolver,
        build_collocated_ns_linear_solvers,
    )
    from fealpy.fvm.collocated_pressure_system import (
        CollocatedPressureSystemControls,
        PressureClosureKind,
    )

    pressure = CollocatedPressureSystemControls()
    solvers = build_collocated_ns_linear_solvers()

    assert (
        pressure.pure_neumann_closure
        is PressureClosureKind.NULLSPACE
    )
    assert isinstance(solvers.momentum, ThirdPartyLinearSolver)
    assert isinstance(
        solvers.pressure_nullspace,
        ThirdPartyLinearSolver,
    )
    assert isinstance(solvers.pressure_dirichlet, FVMLinearSolver)
    assert isinstance(solvers.pressure_gauge, FVMLinearSolver)
    with pytest.raises(ValueError, match="pure_neumann_closure"):
        CollocatedPressureSystemControls(pure_neumann_closure="pin_first_cell")


def test_steady_ns_high_accuracy_profile_has_unique_snapshot():
    from fealpy.fvm.steady_ns_solver_profiles import (
        steady_ns_high_accuracy_simple_profile,
        steady_traction_mms_simple_profile,
    )
    from fealpy.fvm.collocated_pressure_system import PressureClosureKind

    profile = steady_ns_high_accuracy_simple_profile()
    traction = steady_traction_mms_simple_profile()

    assert (
        profile.pressure_system.pure_neumann_closure
        is PressureClosureKind.NULLSPACE
    )
    assert profile.iteration.max_iterations == 1500
    assert profile.iteration.mass_relative_tolerance == pytest.approx(1.0e-7)
    assert profile.momentum_linear_solver.relative_tolerance == pytest.approx(
        1.0e-9
    )
    assert profile.momentum_linear_solver.true_residual_tolerance == pytest.approx(
        1.0e-8
    )
    assert traction.discretization == profile.discretization
    assert (
        traction.momentum_linear_solver
        == profile.momentum_linear_solver
    )
    assert (
        traction.pressure_nullspace_linear_solver
        == profile.pressure_nullspace_linear_solver
    )
    assert (
        traction.pressure_system.pure_neumann_closure
        is PressureClosureKind.GAUGE
    )
    assert traction.iteration.max_iterations == 1000
    assert traction.iteration.mass_relative_tolerance == pytest.approx(1.0e-8)
