"""

Notes
-----
任给一个或者一组重心坐标点，计算出网格中每个单元上对应的笛卡尔坐标。

(1/3, 1/3, 1/3)   --->  (NC, 2)

"""

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

NC = mesh.number_of_cells() # NC = 2


bc = np.array([1/3, 1/3, 1/3], dtype=mesh.ftype)

qf = mesh.integrator(4)
bcs, ws = qf.quadpts, qf.weights

print("bcs:\n", bcs)
print("ws:\n", ws)

area = mesh.entity_measure("cell")
ps = mesh.bc_to_point(bcs) # (NQ, NC, 2)
print(ps)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps, markersize=100)
plt.show()
