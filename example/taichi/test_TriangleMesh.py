import numpy as np
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from fealpy.ti import TriangleMesh

ti.init()


node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=2, ny=2, meshtype='tri', returnnc=True)

mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

node = mesh.entity('cell')
cell = mesh.entity('node')

S0 = ti.field(ti.f64, (NC, 3, 3))

mesh.cell_stiff_matrix(S0);

print(S0)




