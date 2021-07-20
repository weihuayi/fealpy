
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.core  import multi_index_matrix1d 
from fealpy.mesh.core  import multi_index_matrix2d 
from fealpy.mesh.core  import multi_index_matrix3d 

from fealpy.mesh import MeshFactory as MF

p = 6 

mesh = MF.one_triangle_mesh(meshtype='equ')
mi = multi_index_matrix2d(p) # ((p+1)*(p+2)/2, 3)
print(mi)
bcs = mi/p
ps = mesh.bc_to_point(bcs).reshape(-1, 2)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=24)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps, showindex=True, markersize=100, fontsize=28)
plt.show()

if False:
    mi1 = multi_index_matrix1d(p)
    print('mi1:\n', mi1/p)
    mi3 = multi_index_matrix3d(p)
    print('mi3:\n', mi3/p)
