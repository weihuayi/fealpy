from fealpy.backend import backend_manager as bm
from fealpy.fvm import MassResidualMetrics
from fealpy.fvm.simple_residual import (
    collocated_mass_metrics,
)
from fealpy.fvm.solver_diagnostics import (
    EquationResidual,
    equation_residual_converged,
    inexact_inner_tolerance,
    normalized_equation_residual,
)


def test_normalized_equation_residual_uses_complete_lhs_rhs_balance():
    lhs = bm.array([3.0, 4.0])
    rhs = bm.array([0.0, 0.0])

    metrics = normalized_equation_residual(lhs, rhs)

    assert metrics == EquationResidual(
        absolute=5.0,
        relative=1.0,
        scale=5.0,
    )


def test_equation_residual_uses_mixed_absolute_relative_tolerance():
    metrics = EquationResidual(
        absolute=3.4e-17,
        relative=2.1e-5,
        scale=1.6e-12,
    )

    assert equation_residual_converged(metrics, rtol=1.0e-5, atol=1.0e-12)
    assert not equation_residual_converged(metrics, rtol=1.0e-5, atol=0.0)


def test_equation_residual_can_use_unrelaxed_physical_scale():
    metrics = normalized_equation_residual(
        bm.array([101.0]),
        bm.array([100.0]),
        normalization_lhs=bm.array([2.0]),
        normalization_rhs=bm.array([1.0]),
        scale_mode="max",
    )

    assert metrics == EquationResidual(
        absolute=1.0,
        relative=0.5,
        scale=2.0,
    )


def test_equation_residual_accepts_cell_integral_norm_weights():
    metrics = normalized_equation_residual(
        bm.array([1.0, 1.0]),
        bm.array([0.0, 0.0]),
        norm_weights=bm.array([1.0, 4.0]),
    )

    assert metrics == EquationResidual(
        absolute=5.0**0.5,
        relative=1.0,
        scale=5.0**0.5,
    )


def test_inexact_inner_tolerance_tightens_with_outer_residual():
    assert inexact_inner_tolerance(1.0e-4, 1.0e-7) == 1.0e-4
    assert inexact_inner_tolerance(1.0e-4, 1.0e-7, 1.0e-2) == 1.0e-4
    assert inexact_inner_tolerance(1.0e-4, 1.0e-7, 1.0e-4) == 1.0e-5
    assert inexact_inner_tolerance(1.0e-4, 1.0e-7, 1.0e-8) == 1.0e-7
    assert inexact_inner_tolerance(1.0e-8, 1.0e-7, 1.0) == 1.0e-8


def test_cell_integral_residual_uses_inverse_cell_volume_weight():
    lhs = bm.array([1.0, 4.0, 2.0, 8.0])
    rhs = bm.zeros_like(lhs)
    cell_measure = bm.array([1.0, 4.0])

    inverse_measure = 1.0 / bm.tile(cell_measure, (2,))
    metrics = normalized_equation_residual(
        lhs,
        rhs,
        norm_weights=inverse_measure,
        scale_mode="max",
    )

    expected = 5.0
    assert abs(metrics.absolute - expected) < 1.0e-14
    assert metrics.relative == 1.0


class OneBoundaryFaceGeometry:
    S_f = bm.array([[1.0, 0.0]])
    cell_measure = bm.array([4.0])
    is_internal = bm.array([False])
    owner = bm.array([0], dtype=bm.int32)
    neighbour = bm.array([0], dtype=bm.int32)

    @staticmethod
    def scatter_face_flux_to_cells(face_flux):
        return bm.array([face_flux[0]])


def test_collocated_mass_metrics_reports_relative_and_maximum_imbalance():
    geometry = OneBoundaryFaceGeometry()
    face_velocity = bm.array([[2.0, 0.0]])

    metrics = collocated_mass_metrics(face_velocity, geometry=geometry)

    assert metrics == MassResidualMetrics(
        relative_l1=1.0,
        relative_l2=1.0,
        divergence_l2=1.0,
        absolute_linf=2.0,
    )
