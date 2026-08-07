from fealpy.fvm.solver_diagnostics import simple_iteration_log_message


def test_simple_iteration_log_message_reports_required_iteration_data():
    message = simple_iteration_log_message(
        simple_iteration=7,
        nonorthogonal_iterations=3,
        pressure_criterion=1.2e-4,
        momentum_residual=3.4e-5,
        mass_residual=2.3e-3,
        pressure_correction=0.42,
    )

    assert "[SIMPLE 7]" in message
    assert "nonorthogonal iterations: 3" in message
    assert "pressure criterion: 1.20e-04" in message
    assert "pressure relax" not in message
    assert "momentum residual: 3.40e-05" in message
    assert "mass residual: 2.30e-03" in message
    assert "pressure correction L2: 4.20e-01" in message
