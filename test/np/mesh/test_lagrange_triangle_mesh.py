import numpy as np
import matplotlib.pyplot as plt

from fealpy.np.mesh.triangle_mesh import TriangleMesh
from fealpy.np.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface, EllipsoidSurface
from fealpy.np.mesh.lagrange_mesh import LagrangeMesh
from fealpy.np.mesh import functional as F

import pytest
import ipdb

def make_sphere_mesh():
    t = (np.sqrt(5) - 1) / 2
    node = np.array([
        [0, 1, t], [0, 1, -t], [1, t, 0], [1, -t, 0],
        [0, -1, -t], [0, -1, t], [t, 0, 1], [-t, 0, 1],
        [t, 0, -1], [-t, 0, -1], [-1, t, 0], [-1, -t, 0]], dtype=np.float64)
    cell = np.array([
        [6, 2, 0], [3, 2, 6], [5, 3, 6], [5, 6, 7],
        [6, 0, 7], [3, 8, 2], [2, 8, 1], [2, 1, 0],
        [0, 1, 10], [1, 9, 10], [8, 9, 1], [4, 8, 3],
        [4, 3, 5], [4, 5, 11], [7, 10, 11], [0, 10, 7],
        [4, 11, 9], [8, 4, 9], [5, 7, 11], [10, 9, 11]], dtype=np.int_)
    mesh = TriangleMesh(node, cell)
    node = mesh.node
    cell = mesh.entity('cell')
    d = np.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2 + node[:, 2] ** 2) - 1
    l = np.sqrt(np.sum(node ** 2, axis=1))
    n = node / l[..., np.newaxis]
    node = node - d[..., np.newaxis] * n
    return TriangleMesh(node, cell)

def test_to_vtk(p=3):
    surface = SphereSurface() # 以原点为球心，1为半径的球

    mesh = make_sphere_mesh()
    lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=p, surface=surface)

    fname = f"test.vtu"
    lmesh.to_vtk(fname=fname)

def test_generate_local_lagrange_edge(p=3):
    surface = SphereSurface() # 以原点为球心，1为半径的球

    mesh = make_sphere_mesh()
    lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=p, surface=surface)

    localEdge = lmesh.localEdge
    
    print("localEdge:", localEdge)


if __name__ == "__main__":
    test_to_vtk()
    #test_generate_local_lagrange_edge()
