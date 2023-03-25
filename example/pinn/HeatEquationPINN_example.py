
import torch
import torch.nn as nn
from torch.optim import Adam

from fealpy.pinn.machine import LearningMachine
from fealpy.pinn.boundary import TFC2dSpaceTimeDirichletBC
from fealpy.pinn import gradient, ISampler

# 定义神经网络
pinn = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

# 定义优化器、采样器、训练器
optimizer = Adam(pinn.parameters(), lr=0.001)
sampler = ISampler(1000, [[0, 1], [0, 2]], requires_grad=True)

s = TFC2dSpaceTimeDirichletBC(pinn)
lm = LearningMachine(s)

# 设置初边值条件
@s.set_initial
def inital(x: torch.Tensor):
    return torch.sin(torch.pi * x)

@s.set_left_edge(x=0.0)
def left_edge(t: torch.Tensor):
    return torch.zeros_like(t)

@s.set_right_edge(x=2.0)
def right_edge(t: torch.Tensor):
    return torch.zeros_like(t)

# 定义 PDE
def heat_equation(p: torch.Tensor, u):
    phi = u(p)
    phi_t, phi_x = gradient(phi, p, create_graph=True, split=True)
    _, phi_xx = gradient(phi_x, p, create_graph=True, split=True)
    return phi_t - 0.1*phi_xx


# 开始训练
for epoch in range(1000):
    optimizer.zero_grad()
    mse_f = lm.loss(sampler, heat_equation)
    mse_f.backward()
    optimizer.step()

    if epoch % 100 == 0:
        with torch.no_grad():
            print(f"Epoch: {epoch}| Loss: {mse_f.data}")


### Draw the result

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

t = np.linspace(0, 1, 20, dtype=np.float32)
x = np.linspace(0, 2, 20, dtype=np.float32)

phi, (mt, mx) = s.meshgrid_mapping(t, x)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(mt, mx, phi, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('phi')
plt.show()
