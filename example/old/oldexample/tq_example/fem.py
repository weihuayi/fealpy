import taichi as ti
import numpy as np
from opt_einsum import contract

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.ti import lagrange_cell_stiff_matrix_1
from fealpy.ti import lagrange_cell_stiff_matrix_p
from fealpy.ti import lagrange_shape_function

from timeit import default_timer as dtimer 

ti.init(arch=ti.cpu)

# 准备网格

p = 2
ldof = 6 
NQ = 3 

domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=1024, ny=1024, meshtype='tri')

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()


space = LagrangeFiniteElementSpace(mesh, p)

qf = mesh.integrator(2)
bcs, ws = qf.quadpts, qf.weights
NQ = qf.number_of_quadrature_points()
R0, R1 = lagrange_shape_function(bcs, p)

print("R0:]:\n", R0)
print("R1:\n", R1)

ldof = R0.shape[-1]

# 全局变量
w = ti.field(ti.float64, (NQ, ))
R = ti.field(ti.float64, (NQ, ldof, 3))
node = ti.field(ti.float64, (NN, 2))
cell = ti.field(ti.int32, (NC, 3)) 

S0 = ti.field(ti.float64, (NC, 3, 3))
S1 = ti.field(ti.float64, (NC, ldof, ldof))

w.from_numpy(ws)
R.from_numpy(R1)
node.from_numpy(mesh.entity('node'))
cell.from_numpy(mesh.entity('cell'))

start = dtimer()
lagrange_cell_stiff_matrix_1(node, cell, S0)
end = dtimer()
print('run lagrange_cell_stiff_matrix_1 with time:', end - start)

start = dtimer()
lagrange_cell_stiff_matrix_p(6, 3, node, cell, w, R, S1)
end = dtimer()
print('run lagrange_cell_stiff_matrix_p with time:', end - start)

start = dtimer()
gphi = mesh.grad_lambda()
area = mesh.entity_measure('cell')
S2 = np.einsum('jkl, jml, j->jkm', gphi, gphi, area, optimize=True)
end = dtimer()
print('run einsum 1 with time:', end - start)

start = dtimer()
gphi = space.grad_basis(bcs)
area = mesh.entity_measure('cell')
S3 = np.einsum('i, ijkl, ijml, j->jkm', ws, gphi, gphi, area, optimize=True)
end = dtimer()
print('run einsum p with time:', end - start)



print(S0.to_numpy() - S2)
print(S1.to_numpy() - S3)


