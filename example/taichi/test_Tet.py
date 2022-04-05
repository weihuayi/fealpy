
import numpy as np
import taichi as ti

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fealpy.ti import TetrahedronMesh
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import TetrahedronMesh as TMesh

ti.init()

#domain = [0, 1, 0, 1, 0, 1]
#node, cell = MF.boxmesh3d(domain, nx=1, ny=1, nz=1, meshtype='tet',
#        returnnc=True)

node = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0]], dtype=np.float64) # 节点坐标，形状为 (NN, 3)
cell = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32) # 构成每个单元的四个点的编号，形状为 (NC, 4)

mesh = TetrahedronMesh(node, cell)


p = 3
mi = mesh.multi_index_matrix(p)
print(mi)

NP = mesh.number_of_global_interpolation_points(p)
ipoints = ti.field(ti.f64, shape=(NP, 3))
mesh.interpolation_points(p, ipoints)


#mesh = MF.boxmesh3d(domain, nx=1, ny=1, nz=1, meshtype='tet')
mesh = TMesh(node, cell)
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes, showedge=True)
mesh.find_node(axes, node=ipoints.to_numpy(), showindex=True)
plt.show()
