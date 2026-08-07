import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace
from fealpy.mesh import QuadrangleMesh, TetrahedronMesh, TriangleMesh
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
    triangles = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    quadrilaterals = np.array([[1, 4, 3, 5]], dtype=np.int32)

    block = MeshBlock(positions=points)
    block.add_sector(EntitySector("tri", triangles), root=True)
    block.add_sector(EntitySector("quad", quadrilaterals), root=True)
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


@pytest.mark.parametrize(
    "mesh",
    [
        TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=1),
        QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=1),
        TetrahedronMesh.from_box(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=1, ny=1, nz=1
        ),
    ],
)
def test_p0_space_uses_one_cell_dof_on_new_elemental_meshes(mesh):
    bm.set_backend("numpy")
    space = ScaledMonomialSpace(mesh, 0)

    nc = mesh.number_of_cells()
    assert space.number_of_global_dofs() == nc
    np.testing.assert_array_equal(
        np.asarray(space.cell_to_dof()),
        np.arange(nc, dtype=np.int64)[:, None],
    )

    selection = np.array([0, nc - 1], dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(space.cell_to_dof(index=selection)),
        selection[:, None],
    )

    points = np.asarray(space.cellbarycenter)[:, None, :]
    basis = np.asarray(space.basis(points))
    assert basis.shape[-1] == 1
    np.testing.assert_allclose(basis, 1.0)


def test_p0_space_aggregates_mixed_cell_sectors_in_global_order():
    bm.set_backend("numpy")
    mesh = _mixed_tri_quad_mesh()
    space = ScaledMonomialSpace(mesh, 0)

    views = mesh.Entities(-1)
    expected_centers = np.concatenate(
        [np.asarray(view.barycenter()) for view in views], axis=0
    )
    expected_measures = np.concatenate(
        [np.asarray(view.measure()) for view in views], axis=0
    )

    assert space.number_of_global_dofs() == 3
    np.testing.assert_allclose(np.asarray(space.cellbarycenter), expected_centers)
    np.testing.assert_allclose(np.asarray(space.cellmeasure), expected_measures)
    np.testing.assert_array_equal(
        np.asarray(space.cell_to_dof()),
        np.arange(3, dtype=np.int64)[:, None],
    )
