
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse import csr_matrix
from fealpy.mesh import TriangleMesh

node = np.array([
    [0.0, 0.0], # 0
    [1.0, 0.0], # 1
    [1.0, 1.0], # 2
    [0.0, 1.0], # 3
    ], dtype=np.float64)

cell = np.array([
    [1, 2, 0], # 0
    [3, 0, 2], # 1
    ], dtype=np.int_)

mesh = TriangleMesh(node, cell)

mesh.uniform_refine(2)
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

node = mesh.entity('node')
cell = mesh.entity('cell')
cm = mesh.entity_measure('cell') # (NC, )

gphi = mesh.grad_lambda() # (NC, 3, 2)

M = np.zeros((NC, 3, 3), dtype=np.float64)



A = np.einsum('i, ijm, ikm -> ijk', cm, gphi, gphi) # (NC, 3, 3 )


# cell.shape == (NC, 3)

I = np.broadcast_to(cell[:, :, None], shape=A.shape) # (NC, 3, 3)
J = np.broadcast_to(cell[:, None, :], shape=A.shape) # (NC, 3, 3)

A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN))


uh = np.zeros(NN, dtype=np.float64)
uh[0] = 1.0
uh[5] = 1.0

fig = plt.figure()
axes = fig.add_subplot()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize= 12)
mesh.find_cell(axes, showindex=True, fontsize= 16)

fig = plt.figure()
axes= fig.add_subplot(projection='3d')
axes.plot_trisurf(node[:, 0], node[:, 1], cell, uh, 
        cmap='rainbow', lw=3, edgecolors='k')
plt.show()



