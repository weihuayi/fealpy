import argparse
import numpy as np
from fealpy.mesh import EdgeMesh 
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        用线性有限元方法计算梁结构的位移 
        """)

parser.add_argument('--modulus',
        default=2000, type=float,
        help='梁单元的杨氏模量, 默认为 2000 ton/mm^2.')

parser.add_argument('--inertia',
        default=118.6e-6, type=float,
        help='梁单元的惯性矩, 默认为 118.6e-6 ton/mm^2.')

args = parser.parse_args()

E = args.modulus
I = args.inertia

# 构造网格
node = np.array([
    [0], [5]], dtype=np.float64)
cell = np.array([
    [0, 1]] , dtype=np.int_)
mesh = EdgeMesh(node, cell)

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 自由度管理
cell = mesh.entity('cell')
cell2dof = np.zeros((cell.shape[0], 2*(GD+1)), dtype=np.int_)
for i in range(GD+1):
    cell2dof[:, i::(GD+1)] = cell + NN*i

# 组装刚度矩阵
l = mesh.cell_length().reshape(-1, 1)
one = np.ones_like(l)
R = np.array([
            [12*one, 6*l, -12*one, 6*l],
            [6*l, 4*l**2, -6*l, 2*l**2],
            [-12*one, -6*l, 12*one, -6*l],
            [6*l, 2*l**2, -6*l, 4*l**2]], dtype=np.float64)
K = np.einsum('ijkl -> kij', R)
K *= E*I
K /=l[:, None]**3
M = np.broadcast_to(cell2dof[:, :, None], shape=K.shape)
V = np.broadcast_to(cell2dof[:, None, :], shape=K.shape)

K = csr_matrix((K.flat, (M.flat, V.flat)), shape=(NN*(GD+1), NN*(GD+1)))

# 右端项组装
f = 2500 # 施加的荷载
F = np.array([ -1/2*l*f, -1/12*l*f, -1/2*l*f, 1/12*l*f], dtype=np.float64)
F = np.einsum('ijk -> ik', F)

# 求解
uh = np.zeros((NN, GD+1), dtype=np.float64)
uh.T.flat = spsolve(K, F)

print(uh)
