import numpy as np
import pytest
from fealpy.mesh import UniformMesh1d

def test_initialization():
    extent = [0, 10]
    h = 0.5
    origin = 1.0
    mesh = UniformMesh1d(extent, h=h, origin=origin)

    assert mesh.extent == extent
    assert mesh.h == h
    assert mesh.origin == origin
    assert mesh.nx == 10
    assert mesh.NC == 10
    assert mesh.NN == 11

def test_geo_dimension():
    mesh = UniformMesh1d([0, 10])
    assert mesh.geo_dimension() == 1

def test_top_dimension():
    mesh = UniformMesh1d([0, 10])
    assert mesh.top_dimension() == 1

def test_number_of_nodes():
    mesh = UniformMesh1d([0, 10])
    assert mesh.number_of_nodes() == 11

def test_number_of_cells():
    mesh = UniformMesh1d([0, 10])
    assert mesh.number_of_cells() == 10

def test_node():
    mesh = UniformMesh1d([0, 10], h=1.0)
    expected_nodes = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    np.testing.assert_array_equal(mesh.node, expected_nodes)

def test_entity():
    mesh = UniformMesh1d([0, 10], h=1.0)
    expected_cells = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])
    np.testing.assert_array_equal(mesh.entity('cell'), expected_cells)

    expected_nodes = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    np.testing.assert_array_equal(mesh.entity('node'), expected_nodes)

def test_entity_barycenter():
    mesh = UniformMesh1d([0, 10], h=1.0)
    expected_barycenters = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
    np.testing.assert_array_equal(mesh.entity_barycenter('cell'), expected_barycenters)

    expected_nodes = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    np.testing.assert_array_equal(mesh.entity_barycenter('node'), expected_nodes)

def test_function():
    mesh = UniformMesh1d([0, 10], h=1.0)
    f_node = mesh.function(etype='node')
    assert f_node.shape == (11,)
    assert np.all(f_node == 0)

def test_cell_location():
    mesh = UniformMesh1d([0, 10], h=1.0)
    ps = np.array([0.5, 1.7, 3.0, 8.9, 9.01])
    idx = mesh.cell_location(ps)
    expect_location = np.array([0, 1, 3, 8, 9]) 
    np.testing.assert_array_equal(mesh.cell_location(ps), expected_locations)


