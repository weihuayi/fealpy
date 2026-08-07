from pathlib import Path
import importlib.util

from fealpy.backend import backend_manager as bm


def load_piso_example():
    path = Path(__file__).parents[2] / "example" / "fvm" / "ns_fvm_cylinder_piso_example.py"
    spec = importlib.util.spec_from_file_location("ns_fvm_cylinder_piso_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_piso_cylinder_runner_writes_standard_outputs(tmp_path: Path):
    bm.set_backend("numpy")
    example = load_piso_example()

    args = example.create_parser().parse_args(
        [
            "--mesh_size",
            "0.18",
            "--cylinder_mesh_size",
            "0.045",
            "--wake_mesh_size",
            "0.09",
            "--time_steps",
            "2",
            "--duration",
            "0",
            "0.02",
            "--n_correctors",
            "2",
            "--vtk_interval",
            "1",
            "--output_dir",
            str(tmp_path),
        ]
    )

    model, outputs = example.run_piso_cylinder(args)

    assert model.mesh.number_of_cells() > 0
    assert outputs["output_dir"] == tmp_path
    assert (tmp_path / "solution.vtu").exists()
    assert (tmp_path / "flow_overview.png").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "residual_history.csv").exists()
    assert (tmp_path / "force_history.csv").exists()
    assert (tmp_path / "strouhal_summary.csv").exists()
    assert (tmp_path / "snapshots" / "solution_000001.vtu").exists()
    assert (tmp_path / "snapshots" / "solution_000002.vtu").exists()


def test_piso_cylinder_parser_accepts_face_weighted_lsq_gradient_method():
    example = load_piso_example()

    args = example.create_parser().parse_args(
        [
            "--pressure_gradient_method",
            "face_weighted_lsq",
            "--velocity_gradient_method",
            "face_weighted_lsq",
        ]
    )

    assert args.pressure_gradient_method == "face_weighted_lsq"
    assert args.velocity_gradient_method == "face_weighted_lsq"


def test_piso_cylinder_uses_patch_dirichlet_velocity_only():
    bm.set_backend("numpy")
    from fealpy.fvm import CylinderFlowCase, NSFVMPISOModel

    case = CylinderFlowCase(
        mesh_size=0.18,
        cylinder_mesh_size=0.045,
        wake_mesh_size=0.09,
        outlet_velocity_policy="zero",
    )
    model = NSFVMPISOModel(
        {
            "pde": case,
            "mesh_type": "improved_tri",
            "duration": (0.0, 0.02),
            "time_steps": 2,
            "n_correctors": 2,
            "boundary_conditions": case.engineering_boundary_conditions,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    selected_faces = (
        model.solver.spatial_face_velocity.boundary
        .dirichlet_operator.faces
    )
    selected_velocity = (
        model.solver.spatial_face_velocity.boundary
        .dirichlet_operator.values
    )
    face_centers = model.mesh.entity_barycenter("face")[selected_faces]

    assert selected_velocity.shape[0] == selected_faces.shape[0]
    assert bool(bm.to_numpy(bm.any(case.is_inlet_boundary(face_centers))))
    assert bool(bm.to_numpy(bm.any(case.is_cylinder_boundary(face_centers))))
    assert not bool(bm.to_numpy(bm.any(case.is_outlet_boundary(face_centers))))


def test_piso_pressure_flux_includes_dirichlet_pressure_outlet():
    bm.set_backend("numpy")
    from fealpy.fvm import CylinderFlowCase, NSFVMPISOModel

    case = CylinderFlowCase(
        mesh_size=0.18,
        cylinder_mesh_size=0.045,
        wake_mesh_size=0.09,
    )
    model = NSFVMPISOModel(
        {
            "pde": case,
            "mesh_type": "improved_tri",
            "duration": (0.0, 0.02),
            "time_steps": 2,
            "n_correctors": 2,
            "boundary_conditions": case.engineering_boundary_conditions,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    a_p = bm.ones(model.NC)
    pressure = bm.ones(model.NC)
    coef = model.solver.pressure_equation.face_response_coefficient(
        a_p
    )
    equation = model.solver.pressure_equation
    flux = equation.orthogonal_flux(pressure, coef)
    flux = flux - equation.nonorthogonal_cross_flux(
        pressure,
        coef,
        interpolation_method=(
            model.solver.controls.pressure_response_interpolation
        ),
        pressure_gradient=(
            model.solver.pressure_system.boundary.gradient.cell_gradient(
                pressure
            )
        ),
        gradient_boundary=(
            model.solver.pressure_system.boundary.gradient.boundary
        ),
    )
    flux = equation.add_dirichlet_flux(
        flux,
        pressure,
        coef,
        model.solver.pressure_system.boundary.dirichlet_operator.faces,
        model.solver.pressure_system.boundary.dirichlet_operator.values,
    )
    boundary_faces = model.mesh.boundary_face_index()
    face_centers = model.mesh.entity_barycenter("face")[boundary_faces]
    outlet_faces = boundary_faces[case.is_outlet_boundary(face_centers)]

    assert outlet_faces.shape[0] > 0
    assert bool(bm.to_numpy(bm.any(bm.abs(flux[outlet_faces]) > 0.0)))


def test_piso_cylinder_open_outlet_short_run_stays_bounded():
    bm.set_backend("numpy")
    from fealpy.fvm import CylinderFlowCase, NSFVMPISOModel

    case = CylinderFlowCase(
        mesh_size=0.18,
        cylinder_mesh_size=0.045,
        wake_mesh_size=0.09,
    )
    model = NSFVMPISOModel(
        {
            "pde": case,
            "mesh_type": "improved_tri",
            "duration": (0.0, 1.5),
            "time_steps": 30,
            "n_correctors": 4,
            "momentum_nonorthogonal_max_iterations": 10,
            "pressure_nonorthogonal_max_iterations": 10,
            "boundary_conditions": case.engineering_boundary_conditions,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    result = model.solve()
    speed = bm.linalg.norm(result.velocity, axis=1)
    boundary_velocity = (
        model.solver.spatial_face_velocity.boundary
        .dirichlet_operator.values
    )
    inlet_speed = bm.max(bm.sqrt(boundary_velocity[:, 0] ** 2 + boundary_velocity[:, 1] ** 2))

    assert bool(bm.to_numpy(bm.all(bm.isfinite(speed))))
    assert float(bm.to_numpy(bm.max(speed))) < 3.0 * float(bm.to_numpy(inlet_speed))


def test_piso_cylinder_history_reports_outlet_backflow_flux(tmp_path: Path):
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import CylinderFlowCase, NSFVMPISOModel
    CylinderPISOHistory = load_piso_example().CylinderPISOHistory

    case = CylinderFlowCase(
        mesh_size=0.18,
        cylinder_mesh_size=0.045,
        wake_mesh_size=0.09,
    )
    model = NSFVMPISOModel(
        {
            "pde": case,
            "mesh_type": "improved_tri",
            "duration": (0.0, 0.02),
            "time_steps": 2,
            "n_correctors": 2,
            "boundary_conditions": case.engineering_boundary_conditions,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    outlet_faces = model.engineering_bc.patch_face_index("outlet")
    assert outlet_faces.shape[0] >= 2

    flux = bm.zeros(model.mesh.number_of_faces())
    flux = bm.set_at(flux, outlet_faces[0], -0.25)
    flux = bm.set_at(flux, outlet_faces[1], 0.10)

    diagnostics = CylinderPISOHistory._outlet_flux_diagnostics(model, case, flux)

    assert diagnostics["outlet_flux_min"] == pytest.approx(-0.25)
    assert diagnostics["outlet_backflow_flux"] == pytest.approx(0.25)
    assert diagnostics["outlet_backflow_face_count"] == 1
    assert diagnostics["outlet_flux_total"] == pytest.approx(-0.15)
