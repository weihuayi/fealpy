import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import EdgeMesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 参数解析
parser = argparse.ArgumentParser(description="""
        用线性有限元方法计算梁结构的位移
        """)

parser.add_argument('--modulus',
                    default=200e9, type=float,
                    help='梁的杨氏模量, 默认为 200e9 ton/mm^2.')

parser.add_argument('--high',
                    default=317, type=float,
                    help='梁的高度, 默认为 317 ton/mm^2.')

parser.add_argument('--area',
                    default=10.3, type=float,
                    help='梁的宽度, 默认为 10.3 ton/mm^2.')

parser.add_argument('--inertia',
                    default=118.6e-6, type=float,
                    help='梁的惯性矩, 默认为 118.6e-6 ton/mm^2.')

args = parser.parse_args()

E = args.modulus
A = args.area
t = args.high
I = args.inertia

# 构造网格
node = np.array([
    [0], [5],[7.5]], dtype=np.float64)
cell = np.array([
    [0, 1],[1,2]] , dtype=np.int_)
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
l = mesh.cell_length()
l1 =l[0]
l2 =l[1]
print(l.shape)

from fealpy.csm.fem import BeamElementStiffnessIntegrator
from fealpy.csm.fem import BeamDiffusionIntegrator

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
scalar_space = LagrangeFESpace(mesh, 1)
tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 2))
bform = BilinearForm(tensor_space)
K1 = BeamElementStiffnessIntegrator(tensor_space,'pure',E, A, I=I, l=l1)
K1 = K1.assembly()
K2 = BeamElementStiffnessIntegrator(tensor_space,'pure',E, A, I=I, l=l2)
K2 = K2.assembly()

# 组装总刚度矩阵
K_1 = np.zeros((6, 6))
K_1[:4, :4] += K1
K_1[2:6, 2:6] += K2
print(K_1)

pi = BeamDiffusionIntegrator(tensor_space, 'pure', E, A=A, I=I, l=l)
KE = pi.assembly(tensor_space)
bform.add_integrator(pi)
K_2 = bform.assembly()
print(K_2.toarray())




# 单元载荷组装
f = 25000 # 向下施加均匀荷载
F1 = np.array([ -1/2*l1*f, -1/12*l1*l1*f, -1/2*l1*f, 1/12*l1*l1*f], dtype=np.float64)
F2 = np.array([ -1/2*l2*f, -1/12*l2*l2*f, -1/2*l2*f, 1/12*l2*l2*f], dtype=np.float64)

# 总载荷组装
F=np.zeros((6,1))
F[:4,0] +=F1
F[2:6,0] +=F2
print(F)

#边界条件处理
D_0=np.array([[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,1,0,0],
        [0, 0, 0,0, 1,0],
        [0,0,0,0,0,1]])
D_1=np.array([[1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0, 0, 0,0, 0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]])
K=D_0@K_1@D_0+D_1
F=D_0@F

# 将矩阵转换为CSR格式
K= csr_matrix(K)

# 求解
uh = np.zeros((NN, GD+1), dtype=np.float64)
uh.T.flat = spsolve(K, F)

print(uh)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

