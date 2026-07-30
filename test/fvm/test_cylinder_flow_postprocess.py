from pathlib import Path
import math
from types import SimpleNamespace

from fealpy.backend import backend_manager as bm


def test_pressure_drop_uses_nearest_probe_points():
    bm.set_backend("numpy")
    from fealpy.fvm.cylinder_flow_postprocess import pressure_drop

    points = bm.array([[0.15, 0.2], [0.25, 0.2], [1.0, 1.0]])
    pressure = bm.array([3.0, 1.25, -9.0])

    result = pressure_drop(
        points,
        pressure,
        upstream_point=(0.15, 0.2),
        downstream_point=(0.25, 0.2),
    )

    assert result["upstream_cell"] == 0
    assert result["downstream_cell"] == 1
    assert result["delta_p"] == 1.75


def test_zero_cylinder_fields_have_zero_force_coefficients():
    bm.set_backend("numpy")
    from fealpy.fvm.cylinder_flow_postprocess import cylinder_force_coefficients

    class FakeCase:
        rho = 1.0
        mu = 1.0
        mean_velocity = 2.0
        radius = 0.5

        def is_cylinder_boundary(self, points):
            return bm.ones(points.shape[0], dtype=bm.bool)

    geometry = SimpleNamespace(
        is_boundary=bm.array([True, True]),
        face_center=bm.array([[0.0, 0.0], [1.0, 0.0]]),
        cell_center=bm.array([[0.0, 0.5], [1.0, 0.5]]),
        owner=bm.array([0, 1], dtype=bm.int64),
        S_f=bm.array([[0.0, -1.0], [0.0, -1.0]]),
    )

    result = cylinder_force_coefficients(
        object(),
        FakeCase(),
        velocity=bm.zeros((2, 2)),
        pressure=bm.zeros(2),
        velocity_gradient=None,
        geometry=geometry,
    )

    assert result["force_x"] == 0.0
    assert result["force_y"] == 0.0
    assert result["drag_coefficient"] == 0.0
    assert result["lift_coefficient"] == 0.0


def test_cylinder_viscous_force_uses_wall_normal_sn_grad_by_default():
    bm.set_backend("numpy")
    from fealpy.fvm.cylinder_flow_postprocess import cylinder_force_coefficients

    class FakeCase:
        rho = 1.0
        mu = 0.5
        mean_velocity = 2.0
        radius = 0.5

        def is_cylinder_boundary(self, points):
            return bm.ones(points.shape[0], dtype=bm.bool)

    geometry = SimpleNamespace(
        is_boundary=bm.array([True]),
        face_center=bm.array([[1.0, 0.0]]),
        cell_center=bm.array([[0.0, 0.0]]),
        owner=bm.array([0], dtype=bm.int64),
        S_f=bm.array([[2.0, 0.0]]),
    )

    result = cylinder_force_coefficients(
        object(),
        FakeCase(),
        velocity=bm.array([[2.0, 0.0]]),
        pressure=bm.zeros(1),
        geometry=geometry,
    )

    assert abs(result["viscous_force_x"] - 8.0 / 3.0) < 1.0e-12
    assert result["viscous_force_y"] == 0.0


def test_strouhal_summary_estimates_periodic_lift_history():
    from fealpy.fvm.cylinder_flow_postprocess import strouhal_summary

    frequency = 0.5
    reference_length = 0.1
    reference_velocity = 0.2
    rows = []
    for i in range(33):
        time = 0.25 * i
        rows.append(
            {
                "time": time,
                "lift_coefficient": math.sin(2.0 * math.pi * frequency * time),
                "drag_coefficient": 2.0,
            }
        )

    result = strouhal_summary(
        rows,
        reference_length=reference_length,
        reference_velocity=reference_velocity,
        start_time=1.0,
    )

    assert result["valid"] is True
    assert abs(result["frequency"] - frequency) < 1.0e-12
    assert abs(result["strouhal_number"] - 0.25) < 1.0e-12
    assert abs(result["lift_amplitude"] - 1.0) < 1.0e-12
    assert abs(result["mean_drag_coefficient"] - 2.0) < 1.0e-12


def test_strouhal_summary_rejects_tiny_lift_ripple():
    from fealpy.fvm.cylinder_flow_postprocess import strouhal_summary

    rows = []
    for i in range(33):
        time = 0.25 * i
        rows.append(
            {
                "time": time,
                "lift_coefficient": 1.0e-6
                * math.sin(2.0 * math.pi * 0.5 * time),
                "drag_coefficient": 2.0,
            }
        )

    result = strouhal_summary(
        rows,
        reference_length=0.1,
        reference_velocity=0.2,
    )

    assert result["valid"] is False
    assert result["lift_amplitude"] < 1.0e-3


def test_write_cylinder_outputs_creates_summary_and_vtu(tmp_path: Path):
    from dataclasses import replace

    bm.set_backend("numpy")
    from fealpy.fvm import (
        CylinderFlowCase,
        NSFVMSimpleModel,
        steady_ns_high_accuracy_simple_profile,
    )
    from fealpy.fvm.cylinder_flow_postprocess import write_cylinder_outputs

    case = CylinderFlowCase(
        mesh_size=0.16,
        cylinder_mesh_size=0.04,
        wake_mesh_size=0.08,
    )
    base = steady_ns_high_accuracy_simple_profile()
    profile = replace(
        base,
        discretization=replace(
            base.discretization,
            face_flux_correction_scheme="none",
            face_flux_quadrature_order=3,
        ),
        iteration=replace(
            base.iteration,
            max_iterations=2,
            momentum_equation_relaxation=0.7,
            momentum_relative_tolerance=1.0e-3,
            mass_relative_tolerance=1.0e-3,
        ),
    )
    model = NSFVMSimpleModel(
        {
            "pde": case,
            "mesh_type": "improved_tri",
            "profile": profile,
            "boundary_conditions": case.engineering_boundary_conditions,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    solve_result = model.solve()

    result = write_cylinder_outputs(
        model,
        case,
        solve_result,
        tmp_path,
        velocity_gradient=(
            model.solver.spatial_face_velocity.boundary.gradient
        ),
        residuals=solve_result.residual_history,
        run_summary={"solver": "NSFVMSimpleModel"},
    )

    assert (tmp_path / "solution.vtu").exists()
    assert (tmp_path / "flow_overview.png").exists()
    assert (tmp_path / "residual_history.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert "force" in result
    assert "pressure_drop" in result
