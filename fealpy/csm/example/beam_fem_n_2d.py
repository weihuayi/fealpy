import argparse
import numpy as np
from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
from fealpy.mesh import EdgeMesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from fealpy.csm.model import ComputationalModel
from fealpy.csm.model import PDEDataManager

from fealpy.csm.fem import BeamElementStiffnessIntegrator
from fealpy.csm.fem import BeamDiffusionIntegrator
from fealpy.csm.fem import BeamSourceIntegrator

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import DirichletBC

example = 'beam2d'
pde = PDEDataManager('beam').get_example(example) 

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
# node = np.array([
#     [0], [5],[7.5]], dtype=np.float64)
# cell = np.array([
#     [0, 1],[1,2]] , dtype=np.int_)
#mesh = EdgeMesh(node, cell)
mesh = pde.init_mesh()

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()


# 组装刚度矩阵
l = mesh.cell_length()
l1 =l[0]
l2 =l[1]
print(l.shape)


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


pi = BeamDiffusionIntegrator(tensor_space, 'pure', E, A=A, I=I, l=l)
KE = pi.assembly(tensor_space)
bform.add_integrator(pi)
K_2 = bform.assembly()




# 单元载荷组装
f = 25000 # 向下施加均匀荷载
F1 = np.array([ -1/2*l1*f, -1/12*l1*l1*f, -1/2*l1*f, 1/12*l1*l1*f], dtype=np.float64)
F2 = np.array([ -1/2*l2*f, -1/12*l2*l2*f, -1/2*l2*f, 1/12*l2*l2*f], dtype=np.float64)
lform = LinearForm(tensor_space)
FE  = BeamSourceIntegrator(tensor_space, 'pure', source=-f, l=l)
lform.add_integrator(BeamSourceIntegrator(tensor_space, 'pure', source=-f, l=l))
F_source = lform.assembly()

# 总载荷组装
F=np.zeros((6))
F[:4] +=F1
F[2:6] +=F2


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
# TODO: 处理Dirichlet边界条件
gdof = tensor_space.number_of_global_dofs()
threshold = bm.zeros(gdof, dtype=bool)
threshold[pde.dirichlet_dof_index()] = True
print(threshold)
uh = tensor_space.function()
print(uh.shape)
bc = DirichletBC(tensor_space, gd=pde.dirichlet, threshold=threshold)
K_2,F_source = bc.apply(K_2, F_source)
print(K_2)
print(F_source)
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

