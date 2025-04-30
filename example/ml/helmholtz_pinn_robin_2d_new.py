
import time
from matplotlib import pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from fealpy.ml.grad import gradient
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler
from fealpy.mesh import TriangleMesh
from fealpy.pde.helmholtz_2d_new import HelmholtzData2d
from fealpy.typing import TensorLike

# 超参数
num_of_point_pde = 200  # 区域内部的采样点数
num_of_point_bc = 100   # 区域边界上的采样点数
lr = 0.01
iteration = 200 
NN = 32  # 隐藏层单元数

# PDE 
k = 1.0   # 波数
pde = HelmholtzData2d(k=k, backend="pytorch")
domain = pde.domain()

# 构建网格和有限元空间
mesh = TriangleMesh.from_box(box=domain, nx=64, ny=64)


# 定义网络层结构
net_1 = nn.Sequential(
    nn.Linear(2, NN, dtype=bm.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=bm.float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=bm.float64),
    nn.Tanh(),
    nn.Linear(NN//4, 1, dtype=bm.float64)
)
net_2 = nn.Sequential(
    nn.Linear(2, NN, dtype=bm.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=bm.float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=bm.float64),
    nn.Tanh(),
    nn.Linear(NN//4, 1, dtype=bm.float64)
)

# 网络实例化
s_1= Solution(net_1)
s_2= Solution(net_2)

# 选择优化器、损失函数、学习率调度器
optim_1 = Adam(s_1.parameters(), lr=lr, betas=(0.9, 0.99))
optim_2 = Adam(s_2.parameters(), lr=lr, betas=(0.9, 0.99))
mse_cost_func = nn.MSELoss(reduction='mean')
scheduler_1 = StepLR(optim_1, step_size=50, gamma=0.9)
scheduler_2 = StepLR(optim_2, step_size=50, gamma=0.9)

# 采样器
samplerpde_1 = ISampler(ranges=domain, requires_grad=True)
samplerpde_2 = ISampler(ranges=domain, requires_grad=True)
bc1, bc2 = domain[0::2], domain[1::2]  # bc1=(-0.5, -0.5), bc2=(0.5, 0.5)
samplerbc_1 = BoxBoundarySampler(p1=bc1, p2=bc2, requires_grad=True)
samplerbc_2 = BoxBoundarySampler(p1=bc1, p2=bc2, requires_grad=True)


# 定义方程区域内部的损失函数
def mse_pde_fun(p: TensorLike) -> TensorLike:

    u = s_1(p) + 1j * s_2(p)
    f = pde.source(p) # 右端项   

    u_x_real, u_y_real = gradient(u.real, p, create_graph=True, split=True)
    u_x_imag, u_y_imag = gradient(u.imag, p, create_graph=True, split=True)
    u_xx_real, _ = gradient(u_x_real, p, create_graph=True, split=True)
    u_xx_imag, _ = gradient(u_x_imag, p, create_graph=True, split=True)
    _, u_yy_real = gradient(u_y_real, p, create_graph=True, split=True)
    _, u_yy_imag = gradient(u_y_imag, p, create_graph=True, split=True)
    u_xx = u_xx_real + 1j * u_xx_imag
    u_yy = u_yy_real + 1j * u_yy_imag

    return u_xx + u_yy + k**2 * u + f


# 定义方程区域边界的损失函数
def mse_bc_fun(p: TensorLike) -> TensorLike:

    u = s_1(p) + 1j * s_2(p)
    x = p[..., 0]
    y = p[..., 1]
    n = bm.zeros_like(p) # 法向量 n
    n[x > bm.abs(y), 0] = 1.0
    n[y > bm.abs(x), 1] = 1.0
    n[x < -bm.abs(y), 0] = -1.0
    n[y < -bm.abs(x), 1] = -1.0

    grad_u_real = gradient(u.real, p, create_graph=True, split=False)
    grad_u_imag = gradient(u.imag, p, create_graph=True, split=False)
    grad_u = grad_u_real + 1j * grad_u_imag

    kappa = bm.tensor(0.0 + 1j * k)
    g = pde.robin(p=p, n=n)

    return (grad_u*n).sum(dim=-1, keepdim=True) + kappa * u - g


# 训练过程
start_time = time.time()
Loss = []
Error_real = []
Error_imag = []

for epoch in range(iteration+1):

    optim_1.zero_grad()
    optim_2.zero_grad()

    # 实部上的采样与损失计算
    spde_1 = samplerpde_1.run(num_of_point_pde)
    sbc_1 = samplerbc_1.run(num_of_point_bc, num_of_point_bc)
    outpde_1 = mse_pde_fun(spde_1)
    outbc_1 = mse_bc_fun(sbc_1)

    # 虚部上的采样与损失计算
    spde_2 = samplerpde_2.run(num_of_point_pde)
    sbc_2 = samplerbc_2.run(num_of_point_bc, num_of_point_bc)
    outpde_2 = mse_pde_fun(spde_2)
    outbc_2 = mse_bc_fun(sbc_2)

    # 区域内部的损失计算，含实部与虚部
    outpde_real = bm.real(outpde_1)
    outpde_imag = bm.imag(outpde_2)
    mse_pde_real = mse_cost_func(outpde_real, bm.zeros_like(outpde_real))
    mse_pde_imag = mse_cost_func(outpde_imag, bm.zeros_like(outpde_imag))

    # 区域边界的损失计算，含实部与虚部
    outbc_real = bm.real(outbc_1)
    outbc_imag = bm.imag(outbc_2)
    mse_bc_real = mse_cost_func(outbc_real, bm.zeros_like(outbc_real))
    mse_bc_imag = mse_cost_func(outbc_imag, bm.zeros_like(outbc_imag))

    # 总损失函数
    loss = 0.5*(0.5 * mse_pde_real + 0.5 * mse_bc_real) +\
            0.5*(0.5* mse_pde_imag + 0.5 * mse_bc_imag)
    
    loss.backward(retain_graph=True)
    optim_1.step()
    scheduler_1.step()
    optim_2.step()
    scheduler_2.step()

    if epoch % 10 == 0:

        error_real = s_1.estimate_error(pde.solution_numpy_real, mesh, coordtype='c')
        error_imag = s_2.estimate_error(pde.solution_numpy_imag, mesh, coordtype='c')

        Error_real.append(error_real)
        Error_imag.append(error_imag)
        Loss.append(loss.detach().numpy())

        print(f"Epoch: {epoch}, Loss: {loss}")
        print(f"Error_real:{error_real}, Error_imag:{error_imag}")
        print('\n')


end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("训练时间为：", training_time, "秒")

# 损失、实部与虚部的误差的变化曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
y_real = range(1, 10*len(Error_real) +1,10)
y_imag = range(1, 10*len(Error_imag) +1,10)
ax1.plot(y_real, Error_real, label='Real Part')
ax1.plot(y_imag, Error_imag, label='Imaginary Part')
ax1.set_ylim(0, 0.2)  # 设置第一个子图的 y 轴范围
ax1.legend()  # 添加图例

y_loss = range(1, 10 * len(Loss) + 1, 10)
ax2.plot(y_loss, Loss, label='Loss')
ax2.set_ylim(0, 5*1e-4)  # 设置第二个子图的 y 轴范围
ax2.legend()  # 添加图例

plt.tight_layout()


# 可视化网格对比图
bc_ = bm.array([1/3, 1/3, 1/3], dtype=bm.float64)
ps = mesh.bc_to_point(bc_)

u_real = bm.real(pde.solution(ps)).detach().numpy()
u_imag = bm.imag(pde.solution(ps)).detach().numpy()
up_real = s_1(ps).detach().numpy()
up_imag = s_2(ps).detach().numpy()

fig, axes = plt.subplots(2, 2)
axes[0, 0].set_title('Real Part of True Solution')
axes[0, 1].set_title('Imag Part of True Solution')
axes[1, 0].set_title("Real Part of Pinn Module's Solution")
axes[1, 1].set_title("Imag Part of Pinn Module's Solution")

mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

plt.show()
