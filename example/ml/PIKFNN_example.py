import time

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.special import bessel_y0
from torch.optim import Adam
from matplotlib import pyplot as plt

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, get_mesh_sampler, ISampler
from fealpy.ml.integral import linf_error
from fealpy.mesh import TriangleMesh

#方程形式
"""
    \Delta u(x,y) + 200 * u(x,y) = 0 ,   (x,y)\in \Omega
    u(x,y) = \sin(10*x + 10*y) ,         (x,y)\in \partial\Omega

"""

#超参数
NN = 400
NS = 400
lr = 0.1
iteration = 3000

class PIKF_layer(nn.Module):
    def __init__(self, source: Tensor) -> None:
        super().__init__()
        self.source = source

    def kernel_func(self, input: torch.Tensor) ->torch.Tensor:
        a = input[:, None, :] - self.source[None, :, :]
        r = torch.sqrt((a[..., 0:1]**2 + a[..., 1:2]**2).view(input.shape[0], -1))
        val = bessel_y0(torch.sqrt(torch.tensor(200)) * r)/(2 * torch.pi)

        return val

    def forward(self, p: Tensor) -> torch.Tensor:
        return self.kernel_func(p)
    
sampler_s = BoxBoundarySampler(int(NS/4), [-2.5, -2.5], [2.5, 2.5], requires_grad=True)
pikf_layer = PIKF_layer(sampler_s.run())

#定义PIKFNN的网络结构
net_PIKFNN = nn.Sequential(
                           pikf_layer,
                           nn.Linear(NS, 1, dtype=torch.float64, bias=False)
                           )

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -1, 1)

net_PIKFNN.apply(init_weights)

#网络实例化
s = Solution(net_PIKFNN)

#定义优化器、采样器、损失函数
optim = Adam(s.parameters(), lr = lr, betas=(0.9, 0.999) )
sampler = BoxBoundarySampler(int(NN/4), [-1, -1], [1, 1], requires_grad=True)
mse_cost_func = nn.MSELoss(reduction='mean')

mesh = TriangleMesh.from_box([-1 ,1, -1, 1], nx=80, ny=80)
sampler_err = get_mesh_sampler(10, mesh)

#真解
def solution(p:torch.Tensor) -> torch.Tensor:

    x = p[...,0:1]
    y = p[...,1:2]

    return torch.sin(10*x + 10*y)

def solution_numpy(p:np.ndarray) -> np.ndarray:

    sol = solution(torch.tensor(p))
    return sol.detach().numpy()

#边界条件
def bc(p:torch.Tensor, u) -> torch.Tensor:
    return u - solution(p)

# 训练过程
start_time = time.time()
Error = []
Loss = []

for epoch in range(iteration):

    optim.zero_grad()
    nodes_on_bc = sampler.run()
    out_bc = bc(nodes_on_bc, s(nodes_on_bc))

    mse_bc = mse_cost_func(out_bc, torch.zeros_like(out_bc))
    loss = mse_bc

    loss.backward(retain_graph=True)
    optim.step()

    if epoch % 20 == 19:

        error = linf_error(s, solution, sampler=sampler_err)
        Loss.append(loss.detach().numpy())
        Error.append(error.detach().numpy())

        print(f"Epoch: {epoch+1}, Loss: {loss}")
        print(f"Error: {error}")
        print('\n')

end_time = time.time()     # 记录结束时间
training_time = end_time - start_time   # 计算训练时间
print("训练时间为：", training_time, "秒")

#可视化
plt.figure()
y = range(1, 20*len(Loss) +1,20)
plt.plot(y, Loss)

plt.figure()
y = range(1, 20*len(Error) +1,20)
plt.plot(y, Error)

bc_ = np.array([1/3, 1/3, 1/3])
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u_na = torch.real(solution(ps)).detach().numpy()
u_PIKFNN = s(ps).detach().numpy()

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=u_na, linewidths=0, aspect=1)
mesh.add_plot(axes[1], cellcolor=u_PIKFNN, linewidths=0, aspect=1)

plt.show()