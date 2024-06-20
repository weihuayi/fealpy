import numpy as np

import taichi as ti

from fealpy.ti.mesh import TriangleMesh

ti.init(arch=ti.cuda)

mesh = TriangleMesh.from_box()

NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()

print(NC)
print(NN)




