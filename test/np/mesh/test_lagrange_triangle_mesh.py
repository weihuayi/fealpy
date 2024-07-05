import numpy as np

from fealpy.np.mesh.triangle_mesh import TriangleMesh
from fealpy.np.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface
from fealpy.np.mesh.lagrange_mesh import LagrangeMesh
import pytest
import ipdb

p = 1

surface = SphereSurface() # 以原点为球心，1为半径的球
 
node, cell = surface.init_mesh(meshtype='tri', returnnc=True)
mesh = TriangleMesh(node,cell)

mesh1 = LagrangeTriangleMesh.from_triangle_mesh(mesh,p=p)

node = mesh.entity('node')
cell = mesh.entity('cell')

fname = f"test.vtu"
mesh.to_vtk(fname=fname)
