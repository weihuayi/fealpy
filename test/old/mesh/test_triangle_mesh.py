import numpy as np
import matplotlib.pyplot as plt
import ipdb
import pytest
from fealpy.mesh.triangle_mesh import TriangleMesh  # 请替换为实际模块名


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

    assert mesh.ds.number_of_nodes() == 3
    assert mesh.ds.number_of_cells() == 1

def test_triangle_mesh_interpolate():
    mesh = TriangleMesh.from_one_triangle()
    mesh.uniform_refine()
    ips = mesh.interpolation_points(4)
    c2p = mesh.cell_to_ipoint(4)

    fig, axes = plt.subplots()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=ips, showindex=True)
    plt.show()


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

def test_lambda():
    mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=16, ny=16)
    flag = mesh.ds.boundary_cell_flag()
    NC = np.sum(flag)
    val = mesh.grad_lambda(index=flag)
    assert val.shape == (NC, 3, 2)
    val = mesh.rot_lambda(index=flag)
    assert val.shape == (NC, 3, 2)


def test_from_ellipsolid_surface():
    # (-90, 90) 
    mesh, U, V = TriangleMesh.from_ellipsoid_surface(10, 100, 
            radius=(4, 2, 1), theta=(np.pi/2, np.pi/2+np.pi/3))

    mesh.vtkview()

    if False:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        mesh.add_plot(axes)
        plt.show()

def dis(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(len(x), dtype=np.float64)
    val[np.abs(y-0.5)<1e-5] = 1
    return val

def test_interplation_weith_HB():
    from fealpy.functionspace import LagrangeFESpace

    mesh = TriangleMesh.from_unit_square()
    space = LagrangeFESpace(mesh, p=1)
    u = space.function()

    # 获取网格单元数
    NC = mesh.number_of_cells()

    H = np.zeros(NC, dtype=np.float64)

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    u[:] = dis(node)
    H = np.sum(u[:][cell], axis=-1)

    cell2dof = mesh.cell_to_ipoint(p=1)
    isMarkedCell = np.abs(np.sum(u[cell2dof], axis=-1)) > 1.5

    data = {'uh': u, 'H': H}
    option = mesh.bisect_options(disp=False, data=data)
    mesh.bisect(isMarkedCell, options=option)

    space = LagrangeFESpace(mesh, p=1)
    Nu = option['data']['uh']
    NH = option['data']['H']


if __name__ == "__main__":
    test_from_ellipsolid_surface()

