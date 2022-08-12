
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh import MeshFactory as MF

p = 6 

#box = [0, 1, 0, 1]
#mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

mesh = MF.one_triangle_mesh(meshtype='equ')

mi = mesh.multi_index_matrix(p=p)

bcs = mi/p

node = mesh.entity('node')
cell = mesh.entity('cell')

# bcs
# node[cell].shape = (NC, 3, 2)
points = np.einsum('ij, cjk->cik', bcs, node[cell])

print(bcs)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=points[0], 
        showindex=True, fontsize=30)
plt.show()




