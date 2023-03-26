import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TrussMesh import TrussMesh

from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        用线性有限元方法计算桁架结构的位移 
        """)

parser.add_argument('--csection',
        default=2000, type=float,
        help='杆单元的横截面积, 默认为 2000 mm^2.')

parser.add_argument('--modulus',
        default=1500, type=float,
        help='杆单元的杨氏模量, 默认为 1500 ton/mm^2.')


args = parser.parse_args()

A = args.csection
E = args.modulus

# 构造网格
node = np.array([
    [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
    [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
    [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
    [-2540, -2540, 0]], dtype=np.float64)
edge = np.array([
    [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
    [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
    [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
    [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
    [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
mesh = EdgeMesh(node, edge)

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NE= mesh.number_of_edges()

# 自由度管理
edge = mesh.entity('edge')
edge2dof = np.zeros((edge.shape[0], 2*GD), dtype=np.int_)
for i in range(GD):
    edge2dof[:, i::GD] = edge + NN*i

# 组装刚度矩阵
l = mesh.edge_length().reshape(-1, 1)
tan = mesh.unit_edge_tangent()

R = np.einsum('ik, im->ikm', tan, tan)
K = np.zeros((NE, GD*2, GD*2), dtype=np.float64)
K[:, :GD, :GD] = R
K[:, -GD:, :GD] = -R
K[:, :GD, -GD:] = -R
K[:, -GD:, -GD:] = R
K *= E*A
K /= l[:, None]

I = np.broadcast_to(edge2dof[:, :, None], shape=K.shape)
J = np.broadcast_to(edge2dof[:, None, :], shape=K.shape)

K = csr_matrix((K.flat, (I.flat, J.flat)), shape=(NN*GD, NN*GD))

# 右端项
shape = (NN, GD)
F = np.zeros(shape, dtype=np.float64)
F[node[..., 2] == 5080] = np.array([0, 900, 0])

# 边界条件处理
uh = np.zeros(shape, dtype=np.float64)
isDDof = np.tile(node[..., 2] < 1e-12, GD)
F = F.T.flat
x = uh.T.flat
F -=K@x
bdIdx = np.zeros(K.shape[0], dtype=np.int_)
bdIdx[isDDof] = 1
Tbd = spdiags(bdIdx, 0, K.shape[0], K.shape[0])
T = spdiags(1-bdIdx, 0, K.shape[0], K.shape[0])
K = T@K@T + Tbd
F[isDDof] = x[isDDof]

# 求解
uh.T.flat = spsolve(K, F)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d') 
mesh.add_plot(axes)

mesh.node += uh
mesh.add_plot(axes, nodecolor='b', edgecolor='m')
plt.show()


