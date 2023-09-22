
import time

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.ml.grad import gradient
from fealpy.ml.modules import BoxDBCSolution
from fealpy.ml.sampler import  ISampler

#超参数
num_of_point_pde = 50
lr = 0.01
iteration = 150
wavenum = float(1)
NN = 64

# 定义网络层结构，用两个网络分别训练实部和虚部。
net_real = nn.Sequential(
    nn.Linear(2, NN, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//4, 1, dtype=torch.float64)
)

net_imag = nn.Sequential(
    nn.Linear(2, NN, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//4, 1, dtype=torch.float64)
)

#分别对实部和虚部封装dirichlet边界条件
s_real = BoxDBCSolution(net_real)
s_real.set_box([-0.5, 0.5, -0.5, 0.5])
@s_real.set_bc
def c_real(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros(x.shape, dtype=torch.complex128)
    val[:] = torch.cos(k*r)/k
    c = torch.complex(torch.cos(k), torch.sin(
        k))/torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))/k
    val -= c*torch.special.bessel_j0(k*r)
    return torch.real(val)

s_imag = BoxDBCSolution(net_imag)
s_imag.set_box([-0.5, 0.5, -0.5, 0.5])
@s_imag.set_bc
def c(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros(x.shape, dtype=torch.complex128)
    val[:] = torch.cos(k*r)/k
    c = torch.complex(torch.cos(k), torch.sin(
        k))/torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))/k
    val -= c*torch.special.bessel_j0(k*r)
    return torch.imag(val)

# 选择优化器和损失函数
optim_real = Adam(s_real.parameters(), lr=lr, betas=(0.9, 0.99))
optim_imag = Adam(s_imag.parameters(), lr=lr, betas=(0.9, 0.99))
mse_cost_func = nn.MSELoss(reduction='mean')

# 采样器
samplerpde_real = ISampler(
    num_of_point_pde, [[-0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
samplerpde_imag = ISampler(
    num_of_point_pde, [[-0.5, 0.5], [-0.5, 0.5]], requires_grad=True)

# 真解
def solution(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros(x.shape, dtype=torch.complex128)
    val[:] = torch.cos(k*r)/k
    c = torch.complex(torch.cos(k), torch.sin(
        k))/torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))/k
    val -= c*torch.special.bessel_j0(k*r)
    return val

def solution_numpy_real(p: NDArray):
    sol = solution(torch.tensor(p))
    ret = torch.real(sol)
    return ret.detach().numpy()

def solution_numpy_imag(p: NDArray):
    sol = solution(torch.tensor(p))
    ret = torch.imag(sol)
    return ret.detach().numpy()


#根据真解求源项
def f(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    solution_x_real, solution_y_real = gradient(torch.real(solution(p)), p, create_graph=True, split=True)
    solution_x_imag, solution_y_imag = gradient(torch.imag(solution(p)), p, create_graph=True, split=True)
    solution_xx_real, _ = gradient(solution_x_real, p, create_graph=True, split=True)
    solution_xx_imag, _ = gradient(solution_x_imag, p, create_graph=True, split=True)
    _, solution_yy_real = gradient(solution_y_real, p, create_graph=True, split=True)
    _, solution_yy_imag = gradient(solution_y_imag, p, create_graph=True, split=True)
    solution_xx = torch.complex(solution_xx_real, solution_xx_imag)
    solution_yy = torch.complex(solution_yy_real, solution_yy_imag)
    val = - solution_xx - solution_yy - k**2*solution(p)
    return val

# 定义pde
def pde_real(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    u = torch.complex(s_real(p), s_imag(p))
    u_x_real, u_y_real = gradient(torch.real(u), p, create_graph=True, split=True)
    u_x_imag, u_y_imag = gradient(torch.imag(u), p, create_graph=True, split=True)
    u_xx_real, _ = gradient(u_x_real, p, create_graph=True, split=True)
    u_xx_imag, _ = gradient(u_x_imag, p, create_graph=True, split=True)
    _, u_yy_real = gradient(u_y_real, p, create_graph=True, split=True)
    _, u_yy_imag = gradient(u_y_imag, p, create_graph=True, split=True)
    u_xx = torch.complex(u_xx_real, u_xx_imag)
    u_yy = torch.complex(u_yy_real, u_yy_imag)
    return torch.real(u_xx + u_yy + k**2*u + f(p))

def pde_imag(p: torch.Tensor) -> torch.Tensor:

    k = torch.tensor(wavenum)
    u = torch.complex(s_real(p), s_imag(p))
    u_x_real, u_y_real = gradient(torch.real(u), p, create_graph=True, split=True)
    u_x_imag, u_y_imag = gradient(torch.imag(u), p, create_graph=True, split=True)
    u_xx_real, _ = gradient(u_x_real, p, create_graph=True, split=True)
    u_xx_imag, _ = gradient(u_x_imag, p, create_graph=True, split=True)
    _, u_yy_real = gradient(u_y_real, p, create_graph=True, split=True)
    _, u_yy_imag = gradient(u_y_imag, p, create_graph=True, split=True)
    u_xx = torch.complex(u_xx_real, u_xx_imag)
    u_yy = torch.complex(u_yy_real, u_yy_imag)

    return torch.imag(u_xx + u_yy + k**2*u + f(p))

# 训练过程
start_time = time.time()
mesh = TriangleMesh.from_box([-0.5 ,0.5, -0.5, 0.5], nx = 320,ny = 320 )
Error_real = []
Error_imag = []

for epoch in range(iteration+1):

    optim_real.zero_grad()
    optim_imag.zero_grad()

    spde_1 = samplerpde_real.run()
    spde_2 = samplerpde_imag.run()

    outpde_real = pde_real(spde_1)
    outpde_imag = pde_imag(spde_2)
    mse_pde_real = mse_cost_func(outpde_real, torch.zeros_like(outpde_real))
    mse_pde_imag = mse_cost_func(outpde_imag, torch.zeros_like(outpde_imag))

    loss = 0.5*mse_pde_real + 0.5*mse_pde_imag

    loss.backward(retain_graph=True)
    optim_real.step()
    optim_imag.step()

    if epoch % 10 == 0:
        error_real = s_real.estimate_error(solution_numpy_real, mesh, coordtype='c')
        error_imag = s_imag.estimate_error(solution_numpy_imag, mesh, coordtype='c')

        Error_real.append(error_real)
        Error_imag.append(error_imag)

        print(f"Epoch: {epoch}, Loss: {loss}")
        print(f"Error_real:{error_real}, Error_imag:{error_imag}")
        print('\n')

end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("训练时间为：", training_time, "秒")

#可视化
y_real = range(1, 10*len(Error_real) +1,10)
y_imag = range(1, 10*len(Error_imag) +1,10)
plt.plot(y_real, Error_real)
plt.plot(y_imag, Error_imag)

bc_ = np.array([1/3, 1/3, 1/3])
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u_real = torch.real(solution(ps)).detach().numpy()
u_imag = torch.imag(solution(ps)).detach().numpy()
up_real = s_real(ps).detach().numpy()
up_imag = s_imag(ps).detach().numpy()

fig, axes = plt.subplots(2, 2)
mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

plt.show()
