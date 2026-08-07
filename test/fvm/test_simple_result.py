from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fealpy.fvm import (
    SimpleIterationResidual,
    SimpleSolveResult,
)


def _residual(iteration=1):
    return SimpleIterationResidual(
        iteration=iteration,
        mass_relative_l2=1.0e-3,
        mass_relative_l1=2.0e-3,
        mass_divergence_l2=3.0e-3,
        mass_imbalance_linf=4.0e-3,
        pressure_correction_l2=5.0e-3,
        momentum_absolute_l2=6.0e-3,
        momentum_relative_l2=7.0e-3,
        pressure_nonorthogonal_iterations=1,
        momentum_nonorthogonal_iterations=2,
        momentum_nonorthogonal_tolerance=1.0e-6,
    )


def _result(**overrides):
    values = {
        "velocity": np.zeros((2, 2)),
        "pressure": np.zeros(2),
        "face_velocity": np.zeros((5, 2)),
        "face_flux": np.zeros(5),
        "residual_history": (_residual(),),
        "termination_reason": "fixed_point_residuals",
    }
    values.update(overrides)
    return SimpleSolveResult(**values)


def test_simple_result_freezes_bindings_and_exact_field_shapes():
    result = _result()

    assert result.velocity.shape == (2, 2)
    assert result.pressure.shape == (2,)
    assert result.face_velocity.shape == (5, 2)
    assert result.face_flux.shape == (5,)
    with pytest.raises(FrozenInstanceError):
        result.outer_iterations = 2
def test_simple_result_derives_termination_state_from_owned_data():
    converged = _result()
    stopped = _result(termination_reason="max_iterations")

    assert converged.converged is True
    assert stopped.converged is False
    assert converged.outer_iterations == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("velocity", np.zeros(2), "velocity must have shape"),
        ("pressure", np.zeros(3), "pressure must have shape"),
        ("face_velocity", np.zeros((5, 3)), "face velocity must have shape"),
        ("face_flux", np.zeros(4), "face flux must have shape"),
    ],
)
def test_simple_result_rejects_inconsistent_field_shapes(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        _result(**{field: value})
