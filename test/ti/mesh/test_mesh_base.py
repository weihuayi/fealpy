import numpy as np
import taichi as ti
import pytest

from fealpy.mesh import TriangleMesh

from fealpy.ti.mesh import MeshDS

ti.init(arch=ti.cuda)

mesh = TriangleMesh.from_box(nx=1, ny=1)

node = mesh.entity('node')
cell = mesh.entity('cell')

NN = mesh.number_of_nodes()

mds = MeshDS(NN, 2)
mds.node = node
mds.cell = cell

print(mds.entity('node'))
print(mds.entity('cell'))

