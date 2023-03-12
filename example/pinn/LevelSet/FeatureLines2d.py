"""
FeatureLines2d
--------------

根据给定的速度场 `velocity_field` 训练生成各物质点的运动轨迹。
并将结果以网络参数保存为 FeatureLines.pth 文件。
"""

import numpy as np
import torch
from torch.nn import Sequential, Linear, Tanh
from torch.optim import Adam

from fealpy.pinn.machine import LearningMachine, Solution
from fealpy.pinn.sampler import ISampler
from fealpy.pinn.grad import grad_of_fts


class InitialValue(Solution):
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        t = p[:, 0:1]
        xy = p[:, 1:3]
        return self.net(p) - self.net(torch.cat([torch.zeros_like(t), xy], dim=1)) + xy


def velocity_field(p: torch.Tensor):
    x = p[..., 0]
    y = p[..., 1]
    u = torch.zeros(p.shape)
    u[..., 0] = torch.sin((np.pi*x))**2 * torch.sin(2*np.pi*y)
    u[..., 1] = -torch.sin((np.pi*y))**2 * torch.sin(2*np.pi*x)
    return u


pinn = Sequential(
    Linear(3, 64),
    Tanh(),
    Linear(64, 32),
    Tanh(),
    Linear(32, 16),
    Tanh(),
    Linear(16, 8),
    Tanh(),
    Linear(8, 4),
    Tanh(),
    Linear(4, 2)
)


optimizer = Adam(pinn.parameters(), lr=0.0001, weight_decay=1e-5)
s = InitialValue(pinn)
lm = LearningMachine(s)

state_dict = torch.load("FeatureLines.pth")
s.load_state_dict(state_dict)

def pde(p: torch.Tensor, u):
    cp = u(p)
    cp_t = grad_of_fts(cp, p, ft_idx=0, create_graph=True)
    return velocity_field(cp) - cp_t

sampler1 = ISampler(10000, [[-0.25, 0.25], [0, 1], [0, 1]], requires_grad=True)


for epoch in range(2000):
    optimizer.zero_grad()
    mse_f = lm.loss(sampler1, pde)
    mse_f.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch: {epoch} | Loss: {mse_f.data}")

torch.save(s.state_dict(), "FeatureLines.pth")

### Draw the result

from matplotlib import pyplot as plt

t = torch.linspace(-0.25, 0.25, 20)
x = torch.linspace(0, 1, 10)
y = torch.linspace(0, 1, 10)

mt, mx, my = torch.meshgrid(t, x, y, indexing='ij')

inputs = torch.cat([
    mt.flatten().reshape(-1, 1),
    mx.flatten().reshape(-1, 1),
    my.flatten().reshape(-1, 1)
], dim=1)
outputs = s.forward(inputs)
for i in range(100):
    data = outputs[i::100, ...].detach()
    plt.plot(data[:, 0], data[:, 1])

plt.scatter(inputs[:100, 1], inputs[:100, 2])
plt.show()
