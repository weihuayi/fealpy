import numpy as np
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from fealpy.ti import TriangleMesh # 基于 Taichi 的三角形网格

ti.init()


node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)

mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

node = mesh.entity('cell')
cell = mesh.entity('node')

K = mesh.stiff_matrix()

print(K.toarray())







