import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TrussMesh import TrussMesh
from TrussSimulator import TrussSimulator

from scipy.sparse.linalg import spsolve

A = 2000 # 横截面积 mm^2
E = 1500 # 弹性模量 ton/mm^2

# 构造网格
d1 = 952.5 # 单位 mm
d2 = 2540
h1 = 5080
h2 = 2540
node = np.array([
    [-d1, 0, h1], [d1, 0, h1], [-d1, d1, h2], [d1, d1, h2],
    [d1, -d1, h2], [-d1, -d1, h2], [-d2, d2, 0], [d2, d2, 0],
    [d2, -d2, 0], [-d2, -d2, 0]], dtype=np.float64)
edge = np.array([
    [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
    [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
    [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
    [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
    [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
mesh = TrussMesh(node, edge)

simulator = TrussSimulator(mesh)

uh = simulator.function()
M = simulator.striff_matix(A, E)
F = simulator.source_vector(np.abs(node[..., 2]) == 5080, force=np.array([0,
    900, 0]))

M, F = simulator.dirichlet_bc(M, F, np.abs(node[..., 2]) < 1e-12)
uh.T.flat[:] = spsolve(M, F)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d') 
mesh.add_plot(axes)

mesh.node += uh
mesh.add_plot(axes, nodecolor='b', edgecolor='m')
plt.show()

