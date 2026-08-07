import sys
from pathlib import Path

from fealpy.backend import bm
from fealpy.mesher.box import Box2d


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLE_DIR))

import triangle_line_walk_on_new_mesh as line_walk


def test_example_source_does_not_use_non_backend_array_library():
    source = (EXAMPLE_DIR / "triangle_line_walk_on_new_mesh.py").read_text()
    forbidden_import = "import " + "num" + "py"
    forbidden_alias = "n" + "p."

    assert forbidden_import not in source
    assert forbidden_alias not in source


def test_box2d_triangulate_builds_new_mesh_with_expected_entities():
    mesh = Box2d(nx=10, ny=10).triangulate()
    data = line_walk.extract_triangle_mesh_data(mesh)

    assert data.positions.shape == (121, 2)
    assert data.tri.shape == (200, 3)
    assert data.tri_to_edge.shape == (200, 3)
    assert data.edge_to_node.shape[1] == 2

    assert int(bm.sum(data.boundary_edge_mask)) == 40


def test_derived_neighbors_match_boundary_counts():
    mesh = Box2d(nx=10, ny=10).triangulate()
    data = line_walk.extract_triangle_mesh_data(mesh)
    edge_to_cells = line_walk.build_edge_to_cells(
        data.tri_to_edge,
        nedge=data.edge_to_node.shape[0],
    )
    neighbors = line_walk.build_cell_neighbors(data.tri_to_edge, edge_to_cells)

    boundary_slots = int(bm.sum(neighbors < 0))
    assert neighbors.shape == (200, 3)
    assert boundary_slots == 40

    for edge_cells in edge_to_cells:
        assert len(edge_cells) in (1, 2)


def test_line_walk_locates_cell_barycenters():
    mesh = Box2d(nx=10, ny=10).triangulate()
    data = line_walk.extract_triangle_mesh_data(mesh)

    for cell_index in (0, 37, 199):
        point = bm.mean(data.positions[data.tri[cell_index]], axis=0)
        result = line_walk.line_walk_locate(mesh, point, start_cell=0)
        oracle_cells = line_walk.brute_force_locate(data.positions, data.tri, point)

        assert result.status in {"inside", "on_edge", "on_vertex"}
        assert result.located_cell in oracle_cells


def test_line_walk_reports_outside_for_external_point():
    mesh = Box2d(nx=10, ny=10).triangulate()
    data = line_walk.extract_triangle_mesh_data(mesh)
    point = bm.asarray([-0.1, 0.5], dtype=bm.float64)

    result = line_walk.line_walk_locate(mesh, point, start_cell=0)

    assert result.status == "outside"
    assert result.located_cell is None
    assert line_walk.brute_force_locate(data.positions, data.tri, point) == []


def test_line_walk_step_records_follow_most_negative_lambda():
    mesh = Box2d(nx=10, ny=10).triangulate()
    data = line_walk.extract_triangle_mesh_data(mesh)
    point = bm.asarray([0.83, 0.71], dtype=bm.float64)

    result = line_walk.line_walk_locate(mesh, point, start_cell=0)

    assert result.status in {"inside", "on_edge", "on_vertex"}
    for step in result.steps:
        lambdas = bm.asarray(step.lambdas, dtype=bm.float64)
        assert step.walk_edge_index == int(bm.argmin(lambdas))
        assert float(lambdas[step.walk_edge_index]) < -1.0e-12
        assert step.global_edge == int(data.tri_to_edge[step.cell, step.schema_local_edge])
