import taichi as ti
import numpy as np
from opt_einsum import contract

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.ti import lagrange_cell_stiff_matrix_1
from fealpy.ti import lagrange_cell_stiff_matrix_k
from fealpy.ti import lagrange_shape_function

from timeit import default_timer as dtimer 

ti.init(arch=ti.cpu)

# 准备网格

p = 4

domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=1024, ny=1024, meshtype='tri')

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()


space = LagrangeFiniteElementSpace(mesh, p)

qf = mesh.integrator(1)
bcs, ws = qf.quadpts, qf.weights
NQ = qf.number_of_quadrature_points()
R0, R1 = lagrange_shape_function(bcs, p)

ldof = R0.shape[-1]

# 全局变量

R = ti.field(ti.float64, (NQ, ldof, 3))
node = ti.field(ti.float64, (NN, 2))
cell = ti.field(ti.int32, (NC, 3)) 

S0 = ti.field(ti.float64, (NC, 3, 3))

R.from_numpy(R1);
node.from_numpy(mesh.entity('node'))
cell.from_numpy(mesh.entity('cell'))

start = dtimer()
lagrange_cell_stiff_matrix(node, cell, S0)
end = dtimer()
print('run 0 with time:', end - start)


start = dtimer()
gphi = mesh.grad_lambda()
area = mesh.entity_measure('cell')
S1 = np.einsum('jkl, jml, j->jkm', gphi, gphi, area, optimize=True)
end = dtimer()
print('run 1 with time:', end - start)

print(S0.to_numpy() - S1)


