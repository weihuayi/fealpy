import numpy as np
import matplotlib.pyplot as plt

#from fealpy.np.mesh.triangle_mesh import TriangleMesh
from fealpy.np.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface
from fealpy.np.mesh.lagrange_mesh import LagrangeMesh
from fealpy.np.mesh import functional as F

import pytest
import ipdb


def test_generate_local_lagrange_edge():
    p = 3

    multiIndex = F.multi_index_matrix(p, TD=2)

    localEdge = np.zeros((3, p+1), dtype=np.int_)
    a2  = np.where(multiIndex[:, 2] == 0)
    a1  = np.where(multiIndex[:, 1] == 0)
    a0  = np.where(multiIndex[:, 0] == 0)

    localEdge[2, :] = np.array(a2)
    localEdge[1, :] = np.flip(np.array(a1))
    localEdge[0, :] = np.array(a0)

    print("localEdge:", localEdge)


p = 1
node = np.array([[0, 0], [1, 0], [0, 1]])
cell = np.array([[0, 1, 2]])

surface = SphereSurface() # 以原点为球心，1为半径的球

mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface, construct=False)

lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=p, surface=surface)

fname = f"test.vtu"
lmesh.to_vtk(fname=fname)

if __name__ == "__main__":
    test_generate_local_lagrange_edge()
