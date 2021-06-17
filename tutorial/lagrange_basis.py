
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import TriangleMesh
from fealpy.mesh.core import multi_index_matrix2d
from fealpy.functionspace import LagrangeFiniteElementSpace

mesh = MF.one_triangle_mesh(meshtype='equ')

space = LagrangeFiniteElementSpace(mesh, p=5)

n = 50
bcs = multi_index_matrix2d(n)/n  # ldof = (n+1)(n+2)/2 
ps = mesh.bc_to_point(bcs).reshape(-1, 2)  

val = space.basis(bcs) # (NQ, 1, ldof)

val = space.grad_basis(bcs) # (NQ, NC, ldof, GD)

val = val[:, 0, 0]

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps, markersize=20)

fig = plt.figure()
axes = fig.add_subplot(projection='3d')
axes.plot_trisurf(ps[:, 0], ps[:, 1], val, 
        linewidth=0.2, antialiased=True, cmap='rainbow')

NN = mesh.number_of_nodes()
node = np.zeros((NN, 3), dtype=np.float64)
node[:, 0:2] = mesh.entity('node')
cell = mesh.entity('cell')
mesh3 = TriangleMesh(node, cell)
mesh3.add_plot(axes, box=[0, 1, 0, 1, -1, 1], showaxis=True)

plt.show()
