
import numpy as np
from fealpy.mesh import MeshFactory as MF

import taichi as ti
from fealpy.ti import TriangleMesh 
from fealpy.ti import TetrahedronMesh
from fealpy.ti import LagrangeFiniteElementSpace

ti.init()

node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)

mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

space = LagrangeFiniteElementSpace(mesh, p=3)

bc = ti.field(self.ftype, shape=(1, 3))
bc[0] = 1/3
bc[1] = 1/3
bc[2] = 1/3
space.test()

print(space.multiIndex)
print(space.geo_dimension())
print(space.top_dimension())
print(space.number_of_local_dofs())

