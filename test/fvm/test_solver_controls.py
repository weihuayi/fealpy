import pytest

from fealpy.fvm import (
    CollocatedPressureSystemControls,
    PisoSolverControls,
    PoissonSolverControls,
    PressureClosureKind,
)


def test_poisson_controls_require_cellwise_p0_space():
    with pytest.raises(ValueError, match="space_degree must be 0"):
        PoissonSolverControls(space_degree=1)


def test_piso_controls_do_not_configure_the_fixed_p0_space():
    assert "space_degree" not in PisoSolverControls.option_names()


def test_piso_solver_controls_default_to_current_recommended_policy():
    from fealpy.fvm import (
        FVMLinearSolver,
        ThirdPartyLinearSolver,
        build_collocated_ns_linear_solvers,
    )

    controls = PisoSolverControls()
    pressure_system = CollocatedPressureSystemControls()
    solvers = build_collocated_ns_linear_solvers()

    assert controls.diffusion_method == "over_relaxed"
    assert controls.gradient_layer_weights == (1.0, 0.25)
    assert controls.gradient_boundary_weight == 1.0
    assert controls.time_steps == 20
    assert controls.momentum_face_interpolation == "average"
    assert controls.pressure_response_interpolation == "average"
    assert controls.rhie_chow_velocity_interpolation == "average"
    assert controls.momentum_nonorthogonal_max_iterations == 50
    assert controls.momentum_nonorthogonal_rtol == 1.0e-5
    assert controls.pressure_nonorthogonal_max_iterations == 50
    assert controls.pressure_nonorthogonal_rtol == 1.0e-5
    assert (
        pressure_system.pure_neumann_closure
        is PressureClosureKind.NULLSPACE
    )
    assert isinstance(solvers.momentum, ThirdPartyLinearSolver)
    assert isinstance(
        solvers.pressure_nullspace,
        ThirdPartyLinearSolver,
    )
    assert isinstance(solvers.pressure_dirichlet, FVMLinearSolver)
    assert controls.momentum_nonorthogonal_atol == 1.0e-12
    assert controls.pressure_nonorthogonal_atol == 1.0e-12


def test_concrete_solver_set_closes_each_third_party_object_once(
    monkeypatch,
):
    from fealpy.fvm import build_collocated_ns_linear_solvers
    from fealpy.fvm.third_party_linear_solver import (
        ThirdPartyLinearSolver,
    )

    solvers = build_collocated_ns_linear_solvers()
    calls = []

    def record_close(self):
        calls.append(id(self))

    monkeypatch.setattr(ThirdPartyLinearSolver, "close", record_close)
    solvers.close()

    assert calls == [
        id(solvers.momentum),
        id(solvers.pressure_nullspace),
    ]


def test_piso_solver_controls_accept_gradient_reconstruction_weights():
    controls = PisoSolverControls(
        gradient_layer_weights=(1.0, 0.05),
        gradient_boundary_weight=0.75,
    )

    assert controls.gradient_layer_weights == (1.0, 0.05)
    assert controls.gradient_boundary_weight == 0.75


def test_piso_solver_controls_accept_complete_diffusion_variant():
    controls = PisoSolverControls(
        diffusion_method="bounded_over_relaxed",
        diffusion_nonorthogonal_eps=0.1,
    )

    assert controls.diffusion_method == "bounded_over_relaxed"
    assert controls.diffusion_nonorthogonal_eps == 0.1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"diffusion_method": "unknown"}, "diffusion_method"),
        ({"diffusion_nonorthogonal_eps": 0.0}, "diffusion_nonorthogonal_eps"),
    ],
)
def test_piso_solver_controls_reject_invalid_diffusion_variant(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PisoSolverControls(**kwargs)


@pytest.mark.parametrize(
    "name",
    ["momentum_nonorthogonal_atol", "pressure_nonorthogonal_atol"],
)
def test_piso_solver_controls_reject_negative_nonorthogonal_atol(name):
    with pytest.raises(ValueError, match=name):
        PisoSolverControls(**{name: -1.0})


def test_piso_controls_reject_invalid_canonical_interpolation():
    with pytest.raises(ValueError, match="pressure_response_interpolation"):
        PisoSolverControls(pressure_response_interpolation="unknown")
