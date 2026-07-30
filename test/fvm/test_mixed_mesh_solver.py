import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    CollocatedPressureSystemControls,
    CollocatedSimpleSolver,
    FVMGeometry,
    PDEBoundaryConditions,
    PressureClosureKind,
    SimpleDiscretizationControls,
    SimpleIterationControls,
    build_collocated_ns_linear_solvers,
    resolve_simple_boundary_conditions,
)
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.view import Mesh


def _mixed_tri_quad_mesh():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    block = MeshBlock(positions=points)
    block.add_sector(
        EntitySector(
            "tri",
            np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
        ),
        root=True,
    )
    block.add_sector(
        EntitySector(
            "quad",
            np.array([[1, 4, 5, 3]], dtype=np.int32),
        ),
        root=True,
    )
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


def test_collocated_simple_zero_solution_on_mixed_tri_quad_mesh():
    bm.set_backend("numpy")
    mesh = _mixed_tri_quad_mesh()

    def zero_velocity(points):
        return bm.zeros_like(points)

    discretization = SimpleDiscretizationControls()
    iteration = SimpleIterationControls(
        max_iterations=1,
        pressure_relaxation=0.3,
        momentum_relative_tolerance=1.0e-12,
        mass_relative_tolerance=1.0e-12,
    )
    pressure_system = CollocatedPressureSystemControls(
        pure_neumann_closure=PressureClosureKind.GAUGE,
    )
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=zero_velocity,
    )
    solver = CollocatedSimpleSolver(
        diffusion_coef=1.0,
        convection_coef=0.0,
        source=zero_velocity,
        boundary_conditions=resolve_simple_boundary_conditions(
            mesh,
            boundary,
            discretization,
            pressure_system,
        ),
        discretization_controls=discretization,
        iteration_controls=iteration,
        linear_solvers=build_collocated_ns_linear_solvers(),
    )

    result = solver.solve()
    velocity = result.velocity
    pressure = result.pressure

    assert velocity.shape == (3, 2)
    assert pressure.shape == (3,)
    assert result.face_velocity.shape == (8, 2)
    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(pressure, 0.0, atol=1.0e-13)


def test_solution_vtu_rejects_mixed_cell_sectors_until_mesh_writer_supports_them(
    tmp_path,
):
    from fealpy.fvm.benchmark_postprocess import write_solution_vtk

    mesh = _mixed_tri_quad_mesh()
    geometry = FVMGeometry(mesh)

    with pytest.raises(RuntimeError, match="mixed cell-sector VTU output"):
        write_solution_vtk(
            mesh,
            bm.zeros((geometry.NC, geometry.GD)),
            bm.zeros(geometry.NC),
            tmp_path / "mixed.vtu",
            geometry=geometry,
        )
