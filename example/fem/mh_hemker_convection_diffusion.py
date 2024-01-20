import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 计算模型
from fealpy.pde.diffusion_convection_reaction import HemkerDCRModel2d as PDE

# 三角形网格
from fealpy.mesh import TriangleMesh

# 拉格朗日有限元空间
from fealpy.functionspace import LagrangeFESpace

# 区域积分子
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem.mh_convection_integrator import ScalarConvectionIntegrator, Cchoice
from fealpy.fem.mh_source_integrator import ScalarSourceIntegrator

# 双线性型
from fealpy.fem import BilinearForm

# 线性型 
from fealpy.fem import LinearForm

from fealpy.fem import DirichletBC

def source(p):
    x = p[...,0]
    y = p[..., 1]
    return 0*x*y
def d1(p):
    eps = 1e-10
    x = p[..., 0]
    y = p[..., 1]
    flag = np.isclose(x, -3.0, atol=1e-12)
    return flag
def d2(p):
    eps = 1e-16
    x = p[..., 0]
    y = p[..., 1]
    flag = x**2+y**2-1<0.0
    return flag
def d(p):
    eps = 1e-16
    x = p[..., 0]
    y = p[..., 1]
    flag = (x**2+y**2-1<0.0) | np.isclose(x, -3.0, atol=1e-12)
    return flag
## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上线性有限元求解对流占优问题
        """)

parser.add_argument('--dcoef',
        default='0.001', type=float,
        help='扩散项系数，默认为 1.0')

parser.add_argument('--nx',
        default=20, type=int,
        help='网格剖分段数，默认为 20 段')

args = parser.parse_args()
dcoef = args.dcoef
nx = args.nx
p = 1

pde = PDE(A=dcoef, b=(1.0,0.0)) 
domain = pde.domain()
ny = nx
h = 0.3
mesh = TriangleMesh.from_domain_distmesh(domain, h, output=False)
NC = mesh.number_of_cells()
space = LagrangeFESpace(mesh, p=p)

#theta = np.pi/3
#b = np.array([np.cos(theta), -np.sin(theta)])
b = np.array([1.0, 0.0])

# 组装扩散矩阵
s = ScalarDiffusionIntegrator(c=dcoef)
a = BilinearForm(space)
a.add_domain_integrator(s)
S = a.assembly()

grad_uh = np.zeros((NC, 2))
k = 20
for i in range(k):

    uh = space.function()

    # 获得参数
    C = Cchoice(mesh, b, grad_uh)

    # 组装对流矩阵
    cc = ScalarConvectionIntegrator(b, C)
    a = BilinearForm(space)
    a.add_domain_integrator(cc)
    CC = a.assembly()

    # 组装右端矩阵     
    bb = ScalarSourceIntegrator(source, C)
    l = LinearForm(space)
    l.add_domain_integrator(bb)
    bb = l.assembly()

    # 边界处理
    #node = mesh.entity('node')
    #isDirichletNode = pde.is_dirichlet_boundary(node)
    #bc = DirichletBC(space, pde.dirichlet, threshold=isDirichletNode)
    #uh = space.function()
    #A, F = bc.apply(-S+CC, bb, uh)
    #uh[:] = spsolve(A, F)
    isBdDof1 = space.is_boundary_dof(threshold=d1)
    isBdDof2 = space.is_boundary_dof(threshold=d2)
    isBdDof = space.is_boundary_dof(threshold=d)
    gdof = space.number_of_global_dofs()
    ipoint = space.interpolation_points()
    uh[isBdDof1] = 0
    uh[isBdDof2] = 1
    F = bb-(-S+CC)@uh
    uh[~isBdDof] = spsolve((-S+CC)[:, ~isBdDof][~isBdDof, :], F[~isBdDof])
    bc = np.array([1/3, 1/3, 1/3])
    grad_uh = space.grad_value(uh, bc) #(NC,2)
    u = np.zeros(uh.shape)
    u[:] = uh
print('计算结果最小值',np.min(uh))
print('计算结果最大值',np.max(uh))


# 绘图
#fig = plt.figure()
#ax1 = fig.add_subplot(111, projection='3d')
#xx = ipoint[..., 0]
#yy = ipoint[..., 1]
#X = xx.reshape(nx+1, ny+1)
#Y = yy.reshape(ny+1, ny+1)
#Z = u.reshape(nx+1, ny+1)
#ax1.plot_wireframe(X, Y, Z, color='black', linewidth=0.7)
#
#ax1.grid(False)
#ax1.set_box_aspect((1, 1, 0.5))
#plt.show()
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, aspect=0.5)
#mesh.find_node(axes, index=isDirichletNode)
mesh.show_function(plt,uh,cmap='rainbow')
plt.show()

