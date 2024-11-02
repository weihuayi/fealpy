import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fealpy.mesh.simple_mesh_generator import boxmesh3d
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye




L = 1
W = 0.2
lam = 1.25
mu = 1
g = 0.4*(W/L)**2

D = np.zeros((6, 6), dtype=np.float)
D[0:3, 0:3] = lam
D[range(3), range(3)] += mu
D[range(6), range(6)] += mu


mesh = boxmesh3d([0, 1, 0, 0.2,  0, 0.2], nx=50, ny=10, nz=10, meshtype='tet')
cell = mesh.entity('cell')

vol = mesh.entity_measure('cell')
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
grad = mesh.grad_lambda()

B = np.zeros((NC, 12, 6), dtype=np.float) # NC X 12 X 6 

B[:, 0:4, [0, 5, 4]] = grad
B[:, 4:8, [5, 1, 3]] = grad
B[:, 8:12, [4, 3,2]] = grad

A = np.einsum('ijk, km, inm, i->ijn', B, D, B, vol)

cell2dof = np.r_['1', cell, cell+NN, cell+2*NN]
ldof = 12 
I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
J = I.swapaxes(-1, -2)
# Construct the stiffness matrix
A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(3*NN, 3*NN))
b = np.zeros(3*NN, dtype=np.float)
np.add.at(b, cell2dof[:, -4:].flat, -0.25*g)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes, threshold=lambda p: p[..., 0] < 0.5)
plt.show()
