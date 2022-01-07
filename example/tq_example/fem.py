import taichi as ti
import numpy as np
from opt_einsum import contract

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.ti import lagrange_cell_stiff_matrix_0
from fealpy.ti import lagrange_cell_stiff_matrix_1

from timeit import default_timer as dtimer 

ti.init(arch=ti.gpu)

GD = 2

domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=4096, ny=4096, meshtype='tri')

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 全局变量

node = ti.field(ti.float64, (NN, 2))
cell = ti.field(ti.int32, (NC, 3)) 

cnode = ti.field(ti.float64, (NC, 3, 2))

S0 = ti.field(ti.float64, (NC, 3, 3))
S1 = ti.field(ti.float64, (NC, 3, 3))

node.from_numpy(mesh.entity('node'))
cell.from_numpy(mesh.entity('cell'))
cnode.from_numpy(mesh.entity('node')[mesh.entity('cell')])

start = dtimer()
lagrange_cell_stiff_matrix_0(node, cell, S0)
end = dtimer()
print('run 0 with time:', end - start)

start = dtimer()
lagrange_cell_stiff_matrix_1(cnode, S1)
end = dtimer()
print('run 1 with time:', end - start)

start = dtimer()
gphi = mesh.grad_lambda()
area = mesh.entity_measure('cell')
#S2 = np.einsum('jkl, jml, j->jkm', gphi, gphi, area, optimize=True)
S2 = contract('jkl, jml, j->jkm', gphi, gphi, area, optimize=True)
end = dtimer()
print('run 2 with time:', end - start)

print(S0.to_numpy() - S1.to_numpy())
print(S0.to_numpy() - S2)


