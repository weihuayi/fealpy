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

@pytest.mark.parametrize("R, r, Nu, Nv", [
    (3, 1, 20, 20),
    (5, 2, 30, 30),
    (4, 1.5, 15, 10),
    (6, 3, 25, 10)
])
def test_from_torus_surface(R, r, Nu, Nv):

    mesh = TriangleMesh.from_torus_surface(R, r, Nu, Nv)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    # Check if the nodes array has the correct shape
    assert node.shape == (Nu * Nv, 3)

    # Check if the cells array has the correct shape
    print(cell.shape)
    assert cell.shape == (2 * Nu * Nv, 3)

    # Check if the torus wraps around properly in the u-direction
    #u_start = node[:Nv, :]
    #u_end = node[-Nv:, :]
    #np.testing.assert_allclose(u_start, u_end, atol=1e-8)

    # Check if the torus wraps around properly in the v-direction
    #v_start = node[np.arange(0, Nu * Nv, Nv), :]
    #v_end = nodes[np.arange(Nv - 1, Nu * Nv, Nv), :]
    #np.testing.assert_allclose(v_start, v_end, atol=1e-8)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    mesh.add_plot(axes)
    plt.show()

if __name__ == "__main__":

    test_from_torus_surface(3, 1, 20, 20)
