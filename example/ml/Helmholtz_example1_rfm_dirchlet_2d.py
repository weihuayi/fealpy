from math import sqrt

import torch
from torch import Tensor, sin
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import RandomFeaturePoUSpace, PoUSin, Cos, RFFunction, Solution
from fealpy.mesh import UniformMesh2d, TriangleMesh

#方程形式

"""
    \Delta u(x,y) + 200 * u(x,y) = 0 ,   (x,y)\in \Omega
    u(x,y) = \sin(10*x + 10*y) ,         (x,y)\in \partial\Omega

"""
k = torch.sqrt(torch.tensor(200))#波数

#真解

def real_solution(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return sin(10 * x + 10 * y)

#边界条件

def boundary(p: Tensor):
    return real_solution(p)

#源项

def source(p: Tensor):
    x = p[:, 0:1]
    return torch.zeros_like(x)

#超参数

EXT = 2
H = 2*1/EXT
Jn = 512

EXTC = 90
HC = 2*1/EXTC

#用网格进行配置点采样

mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(-1, -1))
node = torch.from_numpy(mesh.entity('node')).clone()
space = RandomFeaturePoUSpace(2, Jn, Cos(), PoUSin(), centers=node, radius=H/2,
                              bound=(k, torch.pi), print_status=True)


mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(-1, -1))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))

mesh_err = TriangleMesh.from_box([-1, 1, -1, 1], nx=50, ny=50)

#计算基函数的值

laplace_phi = space.L(col_in) / sqrt(col_in.shape[0])
phi_in = space.U(col_in) / sqrt(col_in.shape[0])
phi_bd = space.U(col_bd) / sqrt(col_bd.shape[0])

#组装矩阵

A_tensor = torch.cat([laplace_phi + k**2 * phi_in,
                      phi_bd], dim=0)
b_tensor = torch.cat([source(col_in) / sqrt(col_in.shape[0]),
                      boundary(col_bd) / sqrt(col_bd.shape[0])], dim=0)


A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

um = spsolve(A.T@A, A.T@b)
solution = RFFunction(space, torch.from_numpy(um))

#计算L2误差

error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
print(f"L-2 error: {error.data}")

# 可视化

from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(131, projection='3d')
solution.add_surface(axes, box=[-1, 1, -1, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('phi')

axes = fig.add_subplot(132, projection='3d')
Solution(real_solution).add_surface(axes, box=[-1, 1, -1, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('phi')

axes = fig.add_subplot(133, projection='3d')
solution.diff(real_solution).add_surface(axes, box=[-1, 1, -1, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('phi')

plt.show()
