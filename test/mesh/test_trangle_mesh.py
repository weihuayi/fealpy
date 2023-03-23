import numpy as np
import pytest
from fealpy.mesh import TriangleMesh  # 请替换为实际模块名


def test_triangle_mesh_init():
    node = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    cell = np.array([[0, 1, 2]], dtype=np.uint64)

    mesh = TriangleMesh(node, cell)

    assert mesh.node.shape == (3, 2)
    assert mesh.ds.cell.shape == (1, 3)
    assert mesh.meshtype == 'tri'
    assert mesh.itype == np.uint64
    assert mesh.ftype == np.float64
    assert mesh.p == 1

    assert mesh.ds.NN == 3
    assert mesh.ds.NC == 1

@pytest.mark.parametrize("vertices, h", [
    ([(0, 0), (1, 0), (1, 1), (0, 1)], 0.1),
    ([(0, 0), (1, 0), (1, 1), (0, 1)], 0.5),
    ([(0, 0), (1, 0), (0.5, 1)], 0.1),
])
def test_from_polygon_gmsh(vertices, h):

    # 使用 from_polygon_gmsh 函数生成三角形网格
    mesh = TriangleMesh.from_polygon_gmsh(vertices, h)
    cm = mesh.entity_measure('cell')

    assert isinstance(mesh, TriangleMesh)
    assert np.all(cm > 0)


