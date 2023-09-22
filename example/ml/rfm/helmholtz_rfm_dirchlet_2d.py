import time
from math import sqrt

import torch
from torch import Tensor, sin
from scipy.linalg import solve
from matplotlib import pyplot as plt

from fealpy.ml.modules import RandomFeatureSpace, Cos, Function
from fealpy.ml.init import fill_
from fealpy.mesh import UniformMesh2d, TriangleMesh

#方程形式
"""

    \Delta u(x,y) + k**2 * u(x,y) = 0 ,                            (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{k**2/2} * x + \sqrt{k**2/2} * y) ,         (x,y)\in \partial\Omega

"""
K = 1000
k = torch.tensor(K, dtype=torch.float64)#波数

#真解
def real_solution(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)
    # return sin(k*x) + cos(k*y)

#边界条件
def boundary(p: Tensor):
    return real_solution(p)

#源项
def source(p: Tensor):
    x = p[:, 0:1]
    return torch.zeros_like(x)

#超参数
Jn = 8

EXTC = 150
HC = 2*1/EXTC

#用网格进行配置点采样
start_time = time.time()

space = RandomFeatureSpace(2, Jn, Cos(), bound=(0, torch.pi))

k0 = torch.sqrt(k**2/2)
fill_(space.frequency[:, 0], [k0, 0.0, k0, k0], dim=0)
fill_(space.frequency[:, 1], [0.0, k0, k0, -k0], dim=0)


mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(-1, -1))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))

mesh_err = TriangleMesh.from_box([-1, 1, -1, 1], nx=50, ny=50)

#计算基函数的值
laplace_phi = space.laplace_basis(col_in) / sqrt(col_in.shape[0])
phi_in = space.basis(col_in) / sqrt(col_in.shape[0])
phi_bd = space.basis(col_bd) / sqrt(col_bd.shape[0])

#组装矩阵

A_tensor = torch.cat([laplace_phi + k**2 * phi_in,
                      phi_bd], dim=0)
b_tensor = torch.cat([source(col_in) / sqrt(col_in.shape[0]),
                      boundary(col_bd) / sqrt(col_bd.shape[0])], dim=0)


A = A_tensor.detach().cpu().numpy()
b = b_tensor.detach().cpu().numpy()

um = solve(A.T@A, A.T@b)
solution = Function(space, 1, torch.from_numpy(um))

#计算L2误差
error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
print(f"L-2 error: {error.item()}")
end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("Time: ", training_time, "s")

# 可视化
fig = plt.figure()

axes = fig.add_subplot(111)
qm = solution.diff(real_solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[200, 200])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff')
fig.colorbar(qm)

plt.show()
