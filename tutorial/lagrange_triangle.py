
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh import MeshFactory as MF

p = 3 

#box = [0, 1, 0, 1]
#mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

mesh = MF.one_triangle_mesh(meshtype='equ')

mi = mesh.multi_index_matrix(p=p)

print(mi)

bcs = mi/p # (NI, 3)

node = mesh.entity('node') # (NN, 2)
cell = mesh.entity('cell') # (NC, 3)
print("node\n", node)
print("cell\n", cell)

# bcs
# node[cell].shape = (NC, 3, 2)
points = np.einsum('ij, cjk->cik', bcs, node[cell])

print(points[0])

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=points[0], 
        showindex=True, fontsize=30)
plt.show()




