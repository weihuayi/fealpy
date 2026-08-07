import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_example(name):
    path = Path(__file__).parents[2] / "example" / "fvm" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("example_name", "model_name"),
    [
        ("ns_fvm_simple_example.py", "NSFVMSimpleModel"),
        ("stokes_fvm_simple_example.py", "StokesFVMSimpleModel"),
    ],
)
def test_simple_manufactured_example_uses_minimal_model_options(
    example_name,
    model_name,
    monkeypatch,
    capsys,
):
    example = load_example(example_name)
    captured = {}

    class FakeModel:
        def __init__(self, options):
            captured["model_options"] = options

        def __str__(self):
            return "FakeModel"

        def solve(self, **kwargs):
            captured["solve_options"] = kwargs
            result = SimpleNamespace(
                converged=True,
                outer_iterations=7,
                termination_reason="fixed_point_residuals",
            )
            captured["result"] = result
            return result

        def compute_error(self, result):
            assert result is captured["result"]
            return 1.0, 2.0, 3.0, 4.0

    monkeypatch.setattr(example, model_name, FakeModel)

    example.main(
        [
            "--mesh-refine",
            "0",
            "--max-iter",
            "1",
            "--tol",
            "1e-4",
            "--relax",
            "0.2",
        ]
    )

    model_options = captured["model_options"]
    assert model_options["pde"] == 1
    assert model_options["mesh_refine"] == 0
    assert model_options["log_level"] == "INFO"
    profile = model_options["profile"]
    assert profile.iteration.max_iterations == 1
    assert profile.iteration.pressure_relaxation == pytest.approx(0.2)
    assert profile.iteration.momentum_equation_relaxation == pytest.approx(0.9)
    assert profile.iteration.momentum_relative_tolerance == pytest.approx(1.0e-4)
    assert profile.iteration.mass_relative_tolerance == pytest.approx(1.0e-4)
    assert profile.momentum_linear_solver.relative_tolerance == pytest.approx(
        1.0e-9
    )
    assert "mesh_type" not in model_options
    assert {
        "pressure_constraint",
        "momentum_solve_strategy",
        "momentum_component_matrix_policy",
        "momentum_nonorthogonal_max_iter",
        "pressure_nonorthogonal_max_iter",
    }.isdisjoint(model_options)
    assert captured["solve_options"] == {}

    output = capsys.readouterr().out
    assert "SIMPLE converged after 7 iterations" in output
    assert "L2 error (w) = 3.0" in output
    assert "L2 error (p) = 4.0" in output


@pytest.mark.parametrize(
    ("example_name", "model_name"),
    [
        ("ns_fvm_simple_example.py", "NSFVMSimpleModel"),
        ("stokes_fvm_simple_example.py", "StokesFVMSimpleModel"),
    ],
)
def test_simple_manufactured_example_forwards_explicit_runtime_overrides(
    example_name,
    model_name,
    monkeypatch,
):
    example = load_example(example_name)
    captured = {}

    class FakeModel:
        def __init__(self, options):
            captured.update(options)

        def __str__(self):
            return "FakeModel"

        def solve(self, **kwargs):
            result = SimpleNamespace(
                converged=False,
                outer_iterations=1,
                termination_reason="max_iterations",
            )
            captured["result"] = result
            return result

        def compute_error(self, result):
            assert result is captured["result"]
            return 0.0, 0.0, 0.0

    monkeypatch.setattr(example, model_name, FakeModel)

    example.main(
        [
            "--mesh-type",
            "uniform_quad",
            "--quiet",
            "--max-iter",
            "1",
        ]
    )

    assert captured["mesh_type"] == "uniform_quad"
    assert captured["log_level"] == "WARNING"
    assert (
        captured["profile"].momentum_linear_solver.relative_tolerance
        == pytest.approx(1.0e-9)
    )
