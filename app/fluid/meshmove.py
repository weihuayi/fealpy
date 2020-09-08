
import numpy as np

from fealpy.mesh import LagrangeQuadrangleMesh

p=2
n=4
dt = 0.0001
maxit = 1000

node = np.array([
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)], dtype=np.float64)

cell = np.array([(0, 1, 2, 3)], dtype=np.int_)

mesh = LagrangeQuadrangleMesh(node, cell, p=p)
mesh.uniform_refine(n=n)

NC = mesh.number_of_cells()


# 设置材料参数
bc = mesh.entity_barycenter('cell')
u = np.zeros(NC, dtype=np.float64)
flag = (bc[:, 0] < 0.5) & (bc[:, 1] > 0.5)
u[flag] = 1
flag = (bc[:, 0] > 0.5) & (bc[:, 1] < 0.5)
u[flag] = 2
flag = (bc[:, 0] > 0.5) & (bc[:, 1] > 0.5)
u[flag] = 3
mesh.celldata['material'] = u

isInNode = ~mesh.ds.boundary_node_flag()
node = mesh.entity('node')
fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)
for i in range(maxit):
    v = node[:, [1, 0]]
    v[:, 0] *= -1
    node += dt*v
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)



