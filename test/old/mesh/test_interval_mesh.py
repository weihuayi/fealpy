import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.mesh.interval_mesh import IntervalMesh

# import ipdb

def test_interval_domain():
    # ipdb.set_trace()
    mesh = IntervalMesh.from_interval_domain([0, 1], nx=10)
    fig = plt.figure("Interval Mesh Test")
    axes = fig.add_subplot(111)
    mesh.add_plot(axes, showaxis=True)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    plt.show()


def test_mesh_boundary():
    from fealpy.mesh import (
        TriangleMesh,
        UniformMesh2d
    )
    fig = plt.figure("Interval Mesh Test")
    axes = fig.add_subplot(121)
    other_mesh = TriangleMesh.from_one_triangle()
    other_mesh.uniform_refine(1)
    mesh = IntervalMesh.from_mesh_boundary(other_mesh)
    mesh.add_plot(axes, showaxis=True)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)

    axes = fig.add_subplot(122)
    other_mesh = UniformMesh2d([0, 8, 0, 8], h=[0.25, 0.25], origin=[-1, -1])
    mesh = IntervalMesh.from_mesh_boundary(other_mesh)
    mesh.add_plot(axes, showaxis=True)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    plt.show()


def test_circle_boundary():
    mesh = IntervalMesh.from_circle_boundary([0, 0], radius=1.45, n=20)
    fig = plt.figure("Interval Mesh Test")
    axes = fig.add_subplot(111)
    mesh.add_plot(axes, showaxis=True)
    mesh.find_node(axes, showindex=True)
    mesh.find_cell(axes, showindex=True)
    plt.show()


class TestDataStructure():
    def test_topology(self):
        mesh = IntervalMesh.from_interval_domain([0, 1], nx=17)
        ds = mesh.ds
        assert mesh.node.shape == (18, 1)
        assert ds.edge.shape == (17, 2)
        assert ds.face.shape == (18, 1)
        assert ds.cell is mesh.ds.edge

        assert ds.face2cell.shape == (18, 4)

    def test_topology_2(self):
        mesh = IntervalMesh.from_circle_boundary([0, 0], radius=1.45, n=15)
        ds = mesh.ds
        assert mesh.node.shape == (15, 2)
        assert ds.edge.shape == (15, 2)
        assert ds.face.shape == (15, 1)
        assert ds.cell is mesh.ds.edge

        assert ds.face2cell.shape == (15, 4)


if __name__ == '__main__':
    test_interval_domain()
    test_mesh_boundary()
    test_circle_boundary()
