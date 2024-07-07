import numpy as np
import matplotlib.pyplot as plt

from fealpy.np.mesh.triangle_mesh import TriangleMesh
from fealpy.np.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface
from fealpy.np.mesh.lagrange_mesh import LagrangeMesh
import pytest
import ipdb

node = np.array([[0, 0], [1, 0], [0, 1]])
cell = np.array([[0, 1, 2]])

mesh = LagrangeTriangleMesh(node, cell, p=2, construct=False)
print(mesh.localLEdge)
print(mesh.localLFace)


"""
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=20)
mesh.find_cell(axes, showindex=True, fontsize=25)
plt.show()


p = 1
surface = SphereSurface() # 以原点为球心，1为半径的球
 
node, cell = surface.init_mesh(meshtype='tri', returnnc=True)
mesh = TriangleMesh(node,cell)

mesh1 = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=p)

node = mesh.entity('node')
cell = mesh.entity('cell')

fname = f"test.vtu"
mesh1.to_vtk(fname=fname)
"""
