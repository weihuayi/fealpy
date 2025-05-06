import time
from matplotlib import pyplot as plt

from fealpy.backend import backend_manager as bm
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler
from fealpy.ml.methods.pinn import Residual
from fealpy.mesh import TriangleMesh
from fealpy.pde.helmholtz_2d import HelmholtzData2d

#设置 pytorch 后端
bm.set_backend('pytorch')

# 超参数
num_of_point_pde = 200  
num_of_point_bc = 100   
lr = 0.01
iteration = 600 
NN = 32  
num_networks = 2  # 定义网络数量

# PDE 
pde = HelmholtzData2d()
k = pde.k
domain = pde.domain()

# 构建网格和有限元空间
mesh = TriangleMesh.from_box(box=domain, nx=64, ny=64)

# 网络工厂函数
def create_network():
    return nn.Sequential(
        nn.Linear(2, NN, dtype=bm.float64),
        nn.Tanh(),
        nn.Linear(NN, NN//2, dtype=bm.float64),
        nn.Tanh(),
        nn.Linear(NN//2, NN//4, dtype=bm.float64),
        nn.Tanh(),
        nn.Linear(NN//4, 1, dtype=bm.float64)
    )

# 创建网络和解决方案实例
networks = [Solution(create_network()) for _ in range(num_networks)]
s_real, s_imag = networks  # 拆分为实部和虚部网络

# 优化器和学习率调度器
optimizer = Adam([{'params': s_real.parameters()}, {'params': s_imag.parameters()}], lr=lr)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
mse_cost_func = nn.MSELoss(reduction='mean')
residual = Residual('helmholtz_2d')

# 共享采样器
sampler_pde = ISampler(ranges=domain, requires_grad=True)
bc1, bc2 = domain[0::2], domain[1::2]
sampler_bc = BoxBoundarySampler(p1=bc1, p2=bc2, requires_grad=True)

# 训练过程
start_time = time.time()
Loss = []
Error_real = []
Error_imag = []

for epoch in range(iteration+1):
    optimizer.zero_grad()
    
    # 统一采样
    spde = sampler_pde.run(num_of_point_pde)
    sbc = sampler_bc.run(num_of_point_bc, num_of_point_bc)
    
    # 统一计算残差
    pde_res = residual.pde_residual(spde, pde, s_real, s_imag)
    bc_res = residual.bc_residual(sbc, pde, s_real, s_imag)
    
    # 分离实部虚部计算损失
    mse_pde = mse_cost_func(pde_res.real, bm.zeros_like(pde_res.real)) + \
              mse_cost_func(pde_res.imag, bm.zeros_like(pde_res.imag))
    
    mse_bc = mse_cost_func(bc_res.real, bm.zeros_like(bc_res.real)) + \
             mse_cost_func(bc_res.imag, bm.zeros_like(bc_res.imag))
    
    # 总损失
    loss = 0.5 * mse_pde + 0.5 * mse_bc
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 10 == 0:
        error_real = s_real.estimate_error(pde.solution_numpy_real, mesh, coordtype='c')
        error_imag = s_imag.estimate_error(pde.solution_numpy_imag, mesh, coordtype='c')
        
        Error_real.append(error_real)
        Error_imag.append(error_imag)
        Loss.append(loss.detach().numpy())
        
        print(f"Epoch: {epoch}, Loss: {loss}")
        print(f"Error_real:{error_real}, Error_imag:{error_imag}")
        print('\n')

end_time = time.time()
print(f"训练时间: {end_time - start_time:.2f} 秒")

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
ps = mesh.bc_to_point(bm.array([1/3, 1/3, 1/3], dtype=bm.float64))

u_real = bm.real(pde.solution(ps)).detach().numpy()
u_imag = bm.imag(pde.solution(ps)).detach().numpy()
up_real = s_real(ps).detach().numpy()
up_imag = s_imag(ps).detach().numpy()

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
