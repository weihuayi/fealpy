import time
from math import sqrt

import torch
from torch import Tensor
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

from fealpy.ml.modules import RandomFeatureSpace, Function, Besselj0
from fealpy.mesh import UniformMesh2d, TriangleMesh

#方程形式
"""

    -\Delta u(x,y) - k**2 * u(x,y) = f ,                            (x,y)\in \Omega
    robinBC ,         (x,y)\in \partial\Omega

"""

K = 1.0
k = torch.tensor(K, dtype=torch.float64)#波数

#真解
def solution(p: Tensor) -> Tensor:

    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros(x.shape, dtype=torch.complex128)
    val[:] = torch.cos(k*r)/k
    c = torch.complex(torch.cos(k), torch.sin(
        k))/torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))/k
    val -= c*torch.special.bessel_j0(k*r)
    return val

def solution_real(p: Tensor) -> Tensor:
    return torch.real(solution(p))

def solution_imag(p: Tensor) -> Tensor:
    return torch.imag(solution(p))

#梯度
def grad(p: Tensor):

    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)

    val = torch.zeros(p.shape, dtype=torch.complex128)
    t0 = torch.sin(k*r)/r
    c = torch.complex(torch.cos(k), torch.sin(k)) / \
        torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))
    t1 = c * torch.special.bessel_j1(k*r)/r
    t2 = t1 - t0
    val[..., 0:1] = t2*x
    val[..., 1:2] = t2*y
    return val

def grad_real(p: Tensor):
    return torch.real(grad(p))

def grad_imag(p: Tensor):
    return torch.imag(grad(p))

#外法线向量
def n(p: Tensor) -> Tensor:

    x = p[..., 0]
    y = p[..., 1]
    n = torch.zeros_like(p,dtype=torch.float64)
    n[x > torch.abs(y), 0] = 1.0
    n[y > torch.abs(x), 1] = 1.0
    n[x < -torch.abs(y), 0] = -1.0
    n[y < -torch.abs(x), 1] = -1.0
    return n

#robin边界边界条件
def boundary_real(p: Tensor) -> Tensor:

    val = (grad_real(p) * n(p)).sum(dim=-1, keepdim=True) - k * solution_imag(p)
    return val

def boundary_imag(p: Tensor) -> Tensor:

    val = (grad_imag(p) * n(p)).sum(dim=-1, keepdim=True) + k * solution_real(p)
    return val

#源项
def source_real(p: torch.Tensor):

    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros((p.shape[0], 1),dtype=torch.float64)
    val[:] = torch.sin(k*r)/r
    return val

def source_imag(p: torch.Tensor):

    val = torch.zeros((p.shape[0], 1),dtype=torch.float64)
    return val

#超参数
Jn = 60
EXTC = 31
HC = 1/EXTC

#用网格进行配置点采样
start_time = time.time()

space = RandomFeatureSpace(2, Jn, Besselj0() , bound=(2 * k, 0))
mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(-0.5, -0.5))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))

#计算基函数的值
laplace_phi = space.laplace_basis(col_in) / sqrt(col_in.shape[0])
phi_in = space.basis(col_in) / sqrt(col_in.shape[0])
gradient_phi_bd = torch.einsum('ijk, ik->ij', space.grad_basis(col_bd), n(col_bd))/\
sqrt(col_bd.shape[0])
phi_bd = space.basis(col_bd) / sqrt(col_bd.shape[0])

#组装矩阵
start_time = time.time()

A_tensor_in_real = torch.cat([-laplace_phi - k**2 * phi_in, \
                              torch.zeros_like(-laplace_phi - k**2 * phi_in, dtype = torch.float64)], dim = 1)
A_tensor_in_imag = torch.cat([torch.zeros_like(-laplace_phi - k**2 * phi_in, dtype = torch.float64), \
                              -laplace_phi - k**2 * phi_in], dim = 1)
A_tensor_bd_real = torch.cat([gradient_phi_bd, -k * phi_bd], dim = 1)
A_tensor_bd_imag = torch.cat([k * phi_bd, gradient_phi_bd], dim = 1)
A_tensor = torch.cat([
                      A_tensor_in_real,
                      A_tensor_bd_real,
                      A_tensor_in_imag,
                      A_tensor_bd_imag
                      ], dim=0)
b_tensor = torch.cat([
                      source_real(col_in) / sqrt(col_in.shape[0]),
                      boundary_real(col_bd) / sqrt(col_bd.shape[0]),
                      source_imag(col_in) / sqrt(col_in.shape[0]),
                      boundary_imag(col_bd) / sqrt(col_bd.shape[0])
                      ], dim=0)

A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())
um= spsolve(A.T@A, A.T@b)
um_real = um[:um.shape[0]//2]
um_imag = um[um.shape[0]//2:]
u_real = Function(space, torch.from_numpy(um_real))
u_imag = Function(space, torch.from_numpy(um_imag))

end_time = time.time()
time_of_computation = end_time - start_time

#用网格计算误差
mesh_err = TriangleMesh.from_box([-0.5, 0.5, -0.5, 0.5], nx=30, ny=30)
error_real = u_real.estimate_error_tensor(solution_real, mesh_err)
error_imag = u_imag.estimate_error_tensor(solution_imag, mesh_err)
print(f"L-2 error: {error_real.item(),error_imag.item()}")
print("Time:", time_of_computation, "s")

# 可视化
fig = plt.figure()
axes = fig.add_subplot(121)
qm = u_real.diff(solution_real).add_pcolor(axes, box=[-0.5, 0.5, -0.5, 0.5], nums=[300, 300])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff_real')
fig.colorbar(qm)
axes = fig.add_subplot(122)
qm = u_imag.diff(solution_imag).add_pcolor(axes, box=[-0.5, 0.5, -0.5, 0.5], nums=[300, 300])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff_imag')
fig.colorbar(qm)

plt.show()
