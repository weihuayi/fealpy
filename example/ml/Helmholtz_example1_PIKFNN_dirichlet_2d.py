import time

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.special import bessel_y0
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, get_mesh_sampler
from fealpy.ml.integral import linf_error
from fealpy.mesh import TriangleMesh

from uniformly_placed import sample_points_on_square
#方程形式
"""
    \Delta u(x,y) + 200 * u(x,y) = 0 ,   (x,y)\in \Omega
    u(x,y) = \sin(10*x + 10*y) ,         (x,y)\in \partial\Omega

"""

#超参数(配置点个数、源点个数、学习率、迭代次数)
NN = 80
NS = 80
lr = 0.1
iteration = 10000
k = torch.tensor(200) #波数

#PIKF层
class PIKF_layer(nn.Module):
    def __init__(self, source_nodes: Tensor) -> None:
        super().__init__()
        self.source_nodes = source_nodes

    def kernel_func(self, input: torch.Tensor) ->torch.Tensor:
        a = input[:, None, :] - self.source_nodes[None, :, :]
        r = torch.sqrt((a[..., 0:1]**2 + a[..., 1:2]**2).view(input.shape[0], -1))
        val = bessel_y0(torch.sqrt(k) * r)/(2 * torch.pi)

        return val

    def forward(self, p: Tensor) -> torch.Tensor:
        return self.kernel_func(p)
    
pikf_layer = PIKF_layer(sample_points_on_square(-2.5, 2.5, NS))#源点在虚假边界上采样

#PIKFNN的网络结构
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

#优化算法、采样器、损失函数、采用学习率衰减策略
optim = Adam(s.parameters(), lr = lr, betas=(0.9, 0.999) )
sampler = BoxBoundarySampler(int(NN/4), [-1, -1], [1, 1], requires_grad=True)
mse_cost_func = nn.MSELoss(reduction='mean')
scheduler = StepLR(optim, step_size=2000, gamma=0.85)

#真解
def solution(p:torch.Tensor) -> torch.Tensor:

    x = p[...,0:1]
    y = p[...,1:2]

    return torch.sin(torch.sqrt(k/2)*x + torch.sqrt(k/2)*y)

#边界条件
def bc(p:torch.Tensor, u) -> torch.Tensor:
    return u - solution(p)

#构建网格用于计算误差
mesh = TriangleMesh.from_box([-1 ,1, -1, 1], nx=100, ny=100)
sampler_err = get_mesh_sampler(10, mesh)

# 训练网络
start_time = time.time()
Error = []
Loss = []
L2_Error = []

nodes_on_bc = sample_points_on_square(-1, 1, NN)

for epoch in range(iteration):

    optim.zero_grad()
    
    out_bc = bc(nodes_on_bc, s(nodes_on_bc))

    mse_bc = mse_cost_func(out_bc, torch.zeros_like(out_bc))
    loss = mse_bc

    loss.backward(retain_graph=True)
    optim.step()
    scheduler.step()

    if epoch % 50 == 49:

        error = linf_error(s, solution, sampler=sampler_err)
        L2_error = torch.sqrt(
                               torch.sum((s(nodes_on_bc) - solution(nodes_on_bc))**2, dim = 0)\
                              /torch.sum(solution(nodes_on_bc)**2, dim = 0)
                             )
        Loss.append(loss.detach().numpy())
        Error.append(error.detach().numpy())
        L2_Error.append(L2_error.detach().numpy())

        print(f"Epoch: {epoch+1}, Loss: {loss}")
        print(f"Error: {error}")
        print(f"L2_Error: {L2_error}")
        print('\n')
    
end_time = time.time()     
training_time = end_time - start_time   
print("训练时间为：", training_time, "秒")

#可视化Loss曲线、误差曲线、PIKFNN数值解和真解图像
plt.figure()
plt.xlabel('Iteration')
plt.ylabel('Loss')
y = range(1, 50*len(Loss) +1,50)
plt.plot(y, Loss)


plt.figure()
plt.xlabel('Iteration')
plt.ylabel('Error')
y = range(1, 50*len(Error) +1,50)
plt.plot(y, Error)

bc_ = np.array([1/3, 1/3, 1/3])
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u = torch.real(solution(ps)).detach().numpy()
u_PIKFNN = s(ps).detach().numpy()

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=u, linewidths=0, aspect=1)
mesh.add_plot(axes[1], cellcolor=u_PIKFNN, linewidths=0, aspect=1)
axes[0].set_title('u')
axes[1].set_title('u_PIKFNN')

plt.show()
