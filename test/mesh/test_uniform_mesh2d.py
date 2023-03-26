import numpy as np
import pytest
from fealpy.mesh import UniformMesh2d

def test_uniform_mesh2d_init():
    extent = (0, 10, 0, 10)
    h = (1.0, 1.0)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64

    mesh = UniformMesh2d(extent, h=h, origin=origin, itype=itype, ftype=ftype)

    assert mesh.extent == extent
    assert mesh.h == h
    assert mesh.origin == origin
    assert mesh.itype == itype
    assert mesh.ftype == ftype


def test_uniform_mesh2d_node():
    extent = (0, 1, 0, 1)
    h = (1.0, 1.0)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64

    mesh = UniformMesh2d(extent, h=h, origin=origin, itype=itype, ftype=ftype)

    expected_nodes = np.array([[0., 0.],
                               [0., 1.],
                               [1., 0.],
                               [1., 1.]])
    node = mesh.node
    np.testing.assert_array_almost_equal(node.reshape(-1, 2), expected_nodes)

def test_uniform_mesh2d_entity_barycenter():
    extent = (0, 1, 0, 1)
    h = (1.0, 1.0)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64

    mesh = UniformMesh2d(extent, h=h, origin=origin, itype=itype, ftype=ftype)

    # Test cell barycenters
    expected_cell_barycenters = np.array([[[0.5, 0.5]]])
    np.testing.assert_array_almost_equal(mesh.entity_barycenter(etype=2), expected_cell_barycenters)

    # Test face/edge barycenters
    expected_x_barycenters = np.array([[0.5, 0.], [0.5, 1.]]) # (2, 2)
    expected_y_barycenters = np.array([[0., 0.5], [1., 0.5]]) # (2, 2) 

    x_barycenters, y_barycenters = mesh.entity_barycenter(etype=1)
    np.testing.assert_array_almost_equal(x_barycenters.reshape(-1, 2), expected_x_barycenters)
    np.testing.assert_array_almost_equal(y_barycenters.reshape(-1, 2), expected_y_barycenters)

    # Test edgex/facex barycenters
    np.testing.assert_array_almost_equal(mesh.entity_barycenter(etype='edgex').reshape(-1, 2), expected_x_barycenters)

    # Test edgey/facey barycenters
    np.testing.assert_array_almost_equal(mesh.entity_barycenter(etype='edgey').reshape(-1, 2), expected_y_barycenters)

    # Test node barycenters
    expected_nodes_barycenters = np.array([[0., 0.],
                               [0., 1.],
                               [1., 0.],
                               [1., 1.]])
    np.testing.assert_array_almost_equal(mesh.entity_barycenter(etype=0).reshape(-1,
        2), expected_nodes_barycenters)

    # Test invalid entity type
    with pytest.raises(ValueError):
        mesh.entity_barycenter(etype='invalid')

def test_uniform_mesh2d_cell_area_and_edge_length():
    extent = (0, 1, 0, 1)
    h = (1.0, 1.0)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64
    mesh = UniformMesh2d(extent, h, origin, itype, ftype)

    cell_area = mesh.cell_area()
    assert np.isclose(cell_area, 1.0)

    edge_lengths = mesh.edge_length()
    assert np.isclose(edge_lengths[0], 1.0)
    assert np.isclose(edge_lengths[1], 1.0)


def test_uniform_mesh2d_function():
    extent = (0, 1, 0, 1)
    h = (1.0, 1.0)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64

    mesh = UniformMesh2d(extent, h, origin, itype, ftype)

    node_func = mesh.function(etype='node')
    assert node_func.shape == (2, 2)
    assert node_func.dtype == ftype

    edge_func = mesh.function(etype='edge')
    assert len(edge_func) == 2
    assert edge_func[0].shape == (1, 2)
    assert edge_func[1].shape == (2, 1)
    assert edge_func[0].dtype == ftype
    assert edge_func[1].dtype == ftype

    edgex_func = mesh.function(etype='edgex')
    assert edgex_func.shape == (1, 2)
    assert edgex_func.dtype == ftype

    edgey_func = mesh.function(etype='edgey')
    assert edgey_func.shape == (2, 1)
    assert edgey_func.dtype == ftype

    cell_func = mesh.function(etype='cell')
    assert cell_func.shape == (1, 1)
    assert cell_func.dtype == ftype

def test_uniform_mesh2d_value():
    extent = (0, 1, 0, 1)
    h = (0.5, 0.5)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64

    mesh = UniformMesh2d(extent, h, origin, itype, ftype)

    f = np.array([[0, 1], [2, 3]], dtype=ftype)

    points = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])

    values = mesh.value(points, f)

    expected_values = np.array([1.5, 3.5, 2.5, 4.5])
        

    assert np.allclose(values, expected_values)

def test_laplace_operator():
    extent = (0, 3, 0, 3)
    h = (1.0, 1.0)
    origin = (0.0, 0.0)
    itype = np.int_
    ftype = np.float64

    mesh = UniformMesh2d(extent, h, origin, itype, ftype)

    # Call the laplace_operator method to get the Laplace operator matrix
    laplace_matrix = mesh.laplace_operator()

    # Create the expected Laplace operator matrix
    expected_matrix_data = [
        4, -1, 0, -1, 0, 0, 0, 0, 0,
        -1, 4, -1, 0, -1, 0, 0, 0, 0,
        0, -1, 4, 0, 0, -1, 0, 0, 0,
        -1, 0, 0, 4, -1, 0, -1, 0, 0,
        0, -1, 0, -1, 4, -1, 0, -1, 0,
        0, 0, -1, 0, -1, 4, 0, 0, -1,
        0, 0, 0, -1, 0, 0, 4, -1, 0,
        0, 0, 0, 0, -1, 0, -1, 4, -1,
        0, 0, 0, 0, 0, -1, 0, -1, 4
    ]
    expected_matrix = np.array(expected_matrix_data).reshape(9, 9)

    # Compare the obtained Laplace operator matrix with the expected one
    # @TODO 给一个正确的测试数据
    # assert np.allclose(laplace_matrix.toarray(), expected_matrix)


if __name__ == '__main__':
    test_uniform_mesh2d_value()
