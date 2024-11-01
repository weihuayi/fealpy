import numpy as np
import argparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 计算模型
from fealpy.pde.mh_diffusion_convection_model import ConvectionDiffusionModel 

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



## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上线性有限元求解对流占优问题
        """)

parser.add_argument('--dcoef',
        default='1e-7', type=float,
        help='扩散项系数，默认为 1e-7')

parser.add_argument('--nx',
        default=20, type=int,
        help='网格剖分段数，默认为 20 段')

args = parser.parse_args()
dcoef = args.dcoef
nx = args.nx
p = 1

pde = ConvectionDiffusionModel() 
domain = pde.domain()
ny = nx
mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
NC = mesh.number_of_cells()
space = LagrangeFESpace(mesh, p=p)

theta = np.pi/3
b = np.array([np.cos(theta), -np.sin(theta)])

# 组装扩散矩阵
s = ScalarDiffusionIntegrator(c=dcoef)
a = BilinearForm(space)
a.add_domain_integrator(s)
S = a.assembly()

grad_uh = np.zeros((NC, 2))
k = 10
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
    bb = ScalarSourceIntegrator(pde.source, C)
    l = LinearForm(space)
    l.add_domain_integrator(bb)
    bb = l.assembly()

    # 边界处理
    isBdDof1 = space.is_boundary_dof(threshold=pde.d1)
    isBdDof2 = space.is_boundary_dof(threshold=pde.d2)
    isBdDof = space.is_boundary_dof(threshold=pde.d)
    gdof = space.number_of_global_dofs()
    ipoint = space.interpolation_points()
    uh[isBdDof1] = 1
    uh[isBdDof2] = 0
    F = bb-(-S+CC)@uh
    uh[~isBdDof] = spsolve((-S+CC)[:, ~isBdDof][~isBdDof, :], F[~isBdDof])
    bc = np.array([1/3, 1/3, 1/3])
    grad_uh = space.grad_value(uh, bc) #(NC,2)
    u = np.zeros(uh.shape)
    u[:] = uh
print('计算结果最小值',np.min(uh))
print('计算结果最大值',np.max(uh))


# 绘图
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
xx = ipoint[..., 0]
yy = ipoint[..., 1]
X = xx.reshape(nx+1, ny+1)
Y = yy.reshape(ny+1, ny+1)
Z = u.reshape(nx+1, ny+1)
ax1.plot_wireframe(X, Y, Z, color='black', linewidth=0.7)

ax1.grid(False)
ax1.set_box_aspect((1, 1, 0.5))
plt.show()
