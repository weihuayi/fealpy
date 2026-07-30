from pathlib import Path
import importlib.util

from fealpy.backend import backend_manager as bm


def load_simple_example():
    path = Path(__file__).parents[2] / "example" / "fvm" / "ns_fvm_cylinder_simple_example.py"
    spec = importlib.util.spec_from_file_location("ns_fvm_cylinder_simple_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cylinder_simple_runner_writes_standard_outputs(tmp_path: Path):
    bm.set_backend("numpy")
    example = load_simple_example()

    args = example.create_parser().parse_args(
        [
            "--mesh_size",
            "0.16",
            "--cylinder_mesh_size",
            "0.04",
            "--wake_mesh_size",
            "0.08",
            "--max_iter",
            "2",
            "--tol",
            "1e-3",
            "--output_dir",
            str(tmp_path),
        ]
    )

    model, outputs = example.run_simple_cylinder(args)

    assert model.mesh.number_of_cells() > 0
    assert outputs["output_dir"] == tmp_path
    assert (tmp_path / "solution.vtu").exists()
    assert (tmp_path / "flow_overview.png").exists()
    assert (tmp_path / "summary.json").exists()
