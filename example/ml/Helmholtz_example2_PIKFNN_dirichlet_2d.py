import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.special import bessel_y0

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import get_mesh_sampler
from fealpy.ml.integral import linf_error
from fealpy.mesh import TriangleMesh

from uniformly_placed import sample_points_on_circle
from Levenberg_Marquardt_algorithm import minimize_levmarq
#方程形式
"""
    \Delta u(x,y) + k**2 * u(x,y) = 0 ,   (x,y)\in \Omega
    u(x,y) = \sin(k*x + k*y) ,         (x,y)\in \partial\Omega

"""

#超参数(配置点个数、源点个数)
NN = 300
NS = 300
k = torch.tensor(100) #波数

#PIKF层
class PIKF_layer(nn.Module):
    def __init__(self, source_nodes: Tensor) -> None:
        super().__init__()
        self.source_nodes = source_nodes

    def kernel_func(self, input: torch.Tensor) ->torch.Tensor:
        a = input[:, None, :] - self.source_nodes[None, :, :]
        r = torch.sqrt((a[..., 0:1]**2 + a[..., 1:2]**2).view(input.shape[0], -1))
        val = bessel_y0(k * r)/(2 * torch.pi)

        return val

    def forward(self, p: Tensor) -> torch.Tensor:
        return self.kernel_func(p)

#对配置点和源点进行采样
source_nodes = sample_points_on_circle(0, 0, 3, NS)#源点在虚假边界上采样
nodes_on_bc = sample_points_on_circle(0.0, 0.0, 1.0 , NN)

#PIKFNN的网络结构
pikf_layer = PIKF_layer(source_nodes)
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

#真解
def solution(p:torch.Tensor) -> torch.Tensor:

    x = p[...,0:1]
    y = p[...,1:2]

    return torch.sin(k*x) + torch.cos(k*y)

#边界条件
def bc(p:torch.Tensor, u) -> torch.Tensor:
    return u - solution(p)

#构建网格用于计算误差
mesh = TriangleMesh.from_unit_circle_gmsh(0.018)
sampler_err = get_mesh_sampler(20, mesh)

#提取网络层参数并更新
weight = net_PIKFNN[1].weight
xs = weight.view(-1,1)
basis = pikf_layer(nodes_on_bc)

def get_y_hat(x):
    
    y_hat = torch.mm(basis,x)
    return y_hat

new_weight = minimize_levmarq(xs, solution(nodes_on_bc), get_y_hat )
net_PIKFNN[1].weight.data = new_weight.view(1, -1)
weight_1 = net_PIKFNN[1].weight
xs_new = weight_1.view(-1,1)

#计算两种误差
L2_error = torch.sqrt(
            torch.sum((basis @ xs_new - solution(nodes_on_bc))**2, dim = 0)\
            /torch.sum(solution(nodes_on_bc)**2, dim = 0)
          )
print(f"L2_error: {L2_error}")
error = linf_error(s, solution, sampler=sampler_err)
print(f"error: {error}")
  
bc_ = np.array([1/3, 1/3, 1/3])
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u = solution(ps).detach().numpy()
u_PIKFNN = s(ps).detach().numpy()

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=u, linewidths=0, aspect=1)
mesh.add_plot(axes[1], cellcolor=u_PIKFNN, linewidths=0, aspect=1)
axes[0].set_title('u')
axes[1].set_title('u_PIKFNN')

plt.show()
