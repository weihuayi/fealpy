#!/usr/bin/python3
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解possion方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=25, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")

args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator, LinearElasticIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator, VectorSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.solver import cg

from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve


p = args.degree
n = args.n

tmr = timer()
next(tmr)

pde = LShapeRSinData() 
mesh = pde.init_mesh(n=5, meshtype='tri') 
node = mesh.entity('node')
tmr.send('网格生成时间')

space= LagrangeFESpace(mesh, p=p)
uh = space.function() # 建立一个有限元函数
tmr.send('有限元空间生成时间')

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source))
A = bform.assembly()
F = lform.assembly()
tmr.send(f'矩组装时间')

gdof = space.number_of_global_dofs()
A, F = DirichletBC(space, gd=pde.solution).apply(A, F)
tmr.send(f'边界处理时间')

uh[:] = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14)
tmr.send(f' cg 求解器时间')

h = bm.sqrt(mesh.entity_measure('cell'))
error = mesh.error(pde.gradient, uh.grad_value, celltype=True)
tmr.send(f'误差计算时间')

bc = bm.array([[1/3, 1/3, 1/3]], dtype=uh.dtype, device=uh.device)
m0 = -bm.log(error/h)
for i in range(10):
    m1 = space.project(m0)
    m0 = m1(bc)
    print(m0.shape)
tmr.send(f'投影时间')


NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
print(f"节点数: {NN}, 单元数: {NC}")
print(f"max error {error.max()}, min error {error.min()}")
print(f"max m1 {m1.max()}, min m1 {m1.min()}")

next(tmr)

@barycentric
def force(bcs, index):
    return -m1.grad_value(bcs, index)

material = LinearElasticMaterial('movingmesh', elastic_modulus=1,
                                 poisson_ratio=0.3)
tspace = TensorFunctionSpace(space, (-1, 2))
bform = BilinearForm(tspace)
bform.add_integrator(LinearElasticIntegrator(material))
A = bform.assembly()
lform = LinearForm(tspace)
lform.add_integrator(VectorSourceIntegrator(force))
b = lform.assembly()
B = b.reshape(-1, 2)
B[[843, 845, 235, 237, 1201, 1211], :] = 0.0

tmr.send(f'组装线性弹性方程时间')

cornerIndex = bm.array([0, 1, 3, 4, 5, 7], dtype=mesh.itype, device=b.device)
isCornerDof = bm.zeros(A.shape[0], dtype=bm.bool, device=b.device) 
isCornerDof = bm.set_at(isCornerDof, 2*cornerIndex, True)
isCornerDof = bm.set_at(isCornerDof, 2*cornerIndex + 1, True)

bc = DirichletBC(tspace, gd=0.0, threshold=isCornerDof)
A, b = bc.apply(A, b)

    

tmr.send(f"角点 D 氏边界处理时间")

bdNodeIdx = mesh.boundary_node_index()
isBdNode = bm.zeros(NN, dtype=bm.bool, device=b.device)
isBdNode = bm.set_at(isBdNode, bdNodeIdx, True)
isBdNode = bm.set_at(isBdNode, cornerIndex, False)

bdNodeIdex, = bm.where(isBdNode)

gdof = A.shape[0]  

domain = pde.domain()
bdNode = node[isBdNode]
nd = bdNode.shape[0] # 非角点边界节点数
n = domain.grad_signed_dist_function(bdNode) # 非角点边界节点的法向量
val = n.reshape(-1) # 非角点边界节点的法向量
I = bm.repeat(range(nd), 2).reshape(-1)
J = bm.stack([2*bdNodeIdex, 2*bdNodeIdex+1], axis=1).reshape(-1)
G = csr_matrix((val, (I, J)), shape=(nd, gdof))
A = A.to_scipy()

A = bmat([[A, G.T], [G, None]], format='csr')
print(A.shape)
F = bm.concatenate([b, bm.zeros(nd, dtype=b.dtype, device=b.device)], axis=0)

du = spsolve(A, F)
du = du[:-nd].reshape(-1, 2)
print(du.shape)

tmr.send(f"边界内部条件处理时间")



tmr.send(f"求解线性弹性方程时间")


cell = mesh.entity('cell')
node = mesh.entity('node')

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
plot = axes.plot_trisurf(node[:, 0], node[:, 1], m1, triangles=cell, cmap='rainbow')
fig.colorbar(plot, ax=axes)

fig = plt.figure()
axes = fig.add_subplot(111)
axes.quiver(node[:, 0], node[:, 1], b[0::2], b[1::2])


fig = plt.figure()
axes = fig.add_subplot(111)
axes.quiver(node[:, 0], node[:, 1], du[:, 0], du[:, 1])

fig = plt.figure()
axes = fig.add_subplot(121)
mesh.add_plot(axes)
mesh.find_node(axes, index=isBdNode, showindex=True)
axes = fig.add_subplot(122)
node += du
mesh.add_plot(axes)
mesh.find_node(axes, index=isBdNode, showindex=True)
plt.show()
