import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.ml.grad import gradient
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import ISampler

#方程形式
"""

    \Delta u(x,y) + k**2 * u(x,y) = 0 ,                            (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{k**2/2} * x + \sqrt{k**2/2} * y) ,         (x,y)\in \partial\Omega

"""

#超参数
num_of_points_in = 50
num_of_points_bd = 250
lr = 0.01
iteration = 3000
k = torch.tensor(1, dtype=torch.float64)
NN = 64

# 定义网络层结构
net = nn.Sequential(
    nn.Linear(2, NN, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=torch.float64),
    nn.Tanh(),
    nn.Linear(NN//2, 1, dtype=torch.float64)
)

# 网络实例化
s = Solution(net)

# 选择优化器和损失函数
optim = Adam(s.parameters(), lr=lr, betas=(0.9, 0.99))
mse_cost_func = nn.MSELoss(reduction='mean')
scheduler = StepLR(optim, step_size=100, gamma=0.8)

# 采样器
sampler_in = ISampler([[-1, 1], [-1, 1]], requires_grad=True)
sampler_bd = ISampler([[-1, 1], [-1, 1]], requires_grad=True)

#真解
def solution(p: torch.Tensor) -> torch.Tensor:

    x = p[..., 0:1]
    y = p[..., 1:2]
    val = torch.sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)
    return val

#定义pde
def pde(p: torch.Tensor) -> torch.Tensor:

    u = s(p)
    u_x1, u_x2 = gradient(u, p, create_graph=True, split=True)
    u_x1x1, _  = gradient(u_x1, p, create_graph=True, split=True)
    _ , u_x2x2 = gradient(u_x2, p, create_graph=True, split=True)
    return u_x1x1 + u_x2x2 + k**2*u

#定义边界条件
def bc(p: torch.Tensor) -> torch.Tensor:

    x = p[..., 0:1]
    y = p[..., 1:2]
    u = s(p)
    val = torch.sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)
    return u - val

# 训练过程
start_time = time.time()
mesh = TriangleMesh.from_box([-1 ,1, -1, 1], nx = 320,ny = 320)
Loss = []
Error= []

for epoch in range(iteration+1):

    optim.zero_grad()
    nodes_in = sampler_in.run(num_of_points_in)
    nodes_bd = sampler_bd.run(num_of_points_bd)
    output_in = pde(nodes_in)
    output_bd = bc(nodes_bd)

    loss = 0.05 * mse_cost_func(output_in, torch.zeros_like(output_in)) + \
           0.95 * mse_cost_func(output_bd, torch.zeros_like(output_bd))
    loss.backward(retain_graph=True)
    optim.step()
    scheduler.step()

    if epoch % 10 == 0:

        error = s.estimate_error_tensor(solution, mesh)
        Error.append(error.detach().numpy())
        Loss.append(loss.detach().numpy())
        print(f"Epoch: {epoch}, Loss: {loss}")
        print(f"Error:{error.item()}")
        print('\n')

end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("训练时间为：", training_time, "秒")

#可视化
fig_1 = plt.figure()
axes = fig_1.add_subplot(121)
plt.xlabel('Iteration')
plt.ylabel('Loss')
y = range(1, 20*len(Loss) +1,20)
plt.plot(y, Loss)

axes = fig_1.add_subplot(122)
plt.xlabel('Iteration')
plt.ylabel('Error')
y = range(1, 20*len(Error) +1,20)
plt.plot(y, Error)

fig_2 = plt.figure()
axes = fig_2.add_subplot(131)
Solution(solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('u')

axes = fig_2.add_subplot(132)
s.add_pcolor(axes, box=[-1, 1, -1, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('u_PINN')

axes = fig_2.add_subplot(133)
qm = s.diff(solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff')
fig_2.colorbar(qm)

plt.show()
