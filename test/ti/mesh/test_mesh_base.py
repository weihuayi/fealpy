import numpy as np
import taichi as ti
import pytest

from fealpy.mesh import TriangleMesh
import fealpy.ti.numpy as tnp

from fealpy.ti.mesh import MeshDS

ti.init(arch=ti.cuda)

mesh = TriangleMesh.from_box(nx=1, ny=1)
node = mesh.entity('node')
cell = mesh.entity('cell')

NN = mesh.number_of_nodes()

mds = MeshDS(NN, 2)
mds.node = node
mds.cell = cell

mds.localEdge = tnp.array([(1, 2), (2, 0), (0, 1)], dtype=ti.i8)
mds.localFace = tnp.array([(1, 2), (2, 0), (0, 1)], dtype=ti.i8)
mds.ccw = tnp.array([0, 1, 2], dtype=ti.i8)
mds.localCell = tnp.array([
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1)], dtype=ti.i8)

mds.construct()

print(mds.entity('node'))
print(mds.entity('cell'))

