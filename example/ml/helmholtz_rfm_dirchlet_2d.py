import time
from math import sqrt

import torch
from torch import Tensor, sin, cos
from torch.nn.init import normal_
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

from fealpy.ml.modules import RandomFeatureSpace, Cos, Function, Solution
from fealpy.mesh import UniformMesh2d, TriangleMesh

#方程形式

"""

    \Delta u(x,y) + k**2 * u(x,y) = 0 ,                            (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{k**2/2} * x + \sqrt{k**2/2} * y) ,         (x,y)\in \partial\Omega

"""
K = 10000
k = torch.tensor(K)#波数

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

Jn = 100

EXTC = 150
HC = 2*1/EXTC

#用网格进行配置点采样
start_time = time.time()

space = RandomFeatureSpace(2, Jn, Cos(), bound=(0, torch.pi))
with torch.no_grad():
    nf = space.number_of_basis()
    # space.frequency[:] = sqrt(K**2/2)
    space.frequency[:nf//4, 0] = sqrt(K**2/2)
    space.frequency[nf//4:nf//4*2, 1] = sqrt(K**2/2)
    space.frequency[nf//4*2:nf//4*3, :] = sqrt(K**2/2)
    space.frequency[nf//4*3:, 0] = sqrt(K**2/2)
    space.frequency[nf//4*3:, 1] = -sqrt(K**2/2)

# normal_(space.frequency, sqrt(K**2/2), 1)
mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(-1, -1))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))

mesh_err = TriangleMesh.from_box([-1, 1, -1, 1], nx=50, ny=50)

#计算基函数的值

laplace_phi = space.basis_laplace(col_in) / sqrt(col_in.shape[0])
phi_in = space.basis_value(col_in) / sqrt(col_in.shape[0])
phi_bd = space.basis_value(col_bd) / sqrt(col_bd.shape[0])

#组装矩阵

A_tensor = torch.cat([laplace_phi + k**2 * phi_in,
                      phi_bd], dim=0)
b_tensor = torch.cat([source(col_in) / sqrt(col_in.shape[0]),
                      boundary(col_bd) / sqrt(col_bd.shape[0])], dim=0)


A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

um = spsolve(A.T@A, A.T@b)
solution = Function(space, torch.from_numpy(um))

#计算L2误差

error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
print(f"L-2 error: {error.item()}")
end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("训练时间为：", training_time, "秒")

# 可视化

fig = plt.figure()

axes = fig.add_subplot(111)
qm = solution.diff(real_solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[200, 200])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff')
fig.colorbar(qm)

plt.show()
