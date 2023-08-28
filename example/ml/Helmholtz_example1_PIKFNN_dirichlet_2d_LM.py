import time

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.special import bessel_y0

from fealpy.ml.modules import Solution
# from fealpy.ml.sampler import get_mesh_sampler
# from fealpy.ml.integral import linf_error
# from fealpy.mesh import TriangleMesh

from uniformly_placed import sample_points_on_square
from Levenberg_Marquardt_algorithm import minimize_levmarq

#方程形式

"""

    \Delta u(x,y) + k^2 * u(x,y) = 0 ,                            (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{k^2/2} * x + \sqrt{k^2/2} * y) ,         (x,y)\in \partial\Omega

"""

#超参数(配置点个数、源点个数、波数)

num_of_points_in = 6000
num_of_points_source = 6000
k = torch.tensor(1000, dtype=torch.float64) 

#PIKF层

class PIKF_layer(nn.Module):
    def __init__(self, source_nodes: Tensor) -> None:
        super().__init__()
        self.source_nodes = source_nodes

    def kernel_func(self, input: torch.Tensor) ->torch.Tensor:
        a = input[:, None, :] - self.source_nodes[None, :, :]
        r = torch.sqrt((a[..., 0:1]**2 + a[..., 1:2]**2).view(input.shape[0], -1))
        val = bessel_y0(k* r)/(2 * torch.pi)

        return val

    def forward(self, p: Tensor) -> torch.Tensor:

        return self.kernel_func(p)
    
pikf_layer = PIKF_layer(sample_points_on_square(-2.5, 2.5, num_of_points_source))#源点在虚假边界上采样

#PIKFNN的网络结构

net_PIKFNN = nn.Sequential(
                           pikf_layer,
                           nn.Linear(num_of_points_source, 1, dtype=torch.float64, bias=False)
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

    return torch.sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)

#边界条件

def bc(p:torch.Tensor, u) -> torch.Tensor:
    return u - solution(p)

# 提取网络参数并更新

start_time = time.time()

nodes_on_bc = sample_points_on_square(-1, 1, num_of_points_in)

weight = net_PIKFNN[1].weight
w = weight.view(-1,1)
basis = pikf_layer(nodes_on_bc)

def get_y_hat(x):
    
    y_hat = torch.mm(basis,x)
    return y_hat

new_weight = minimize_levmarq(w, solution(nodes_on_bc), get_y_hat, max_iters=30 )
net_PIKFNN[1].weight.data = new_weight.view(1, -1)
del new_weight
weight_1 = net_PIKFNN[1].weight
xs_new = weight_1.view(-1,1)
del weight_1

#计算L2相对误差

L2_error = torch.sqrt(
            torch.sum((s(nodes_on_bc) - solution(nodes_on_bc))**2, dim = 0)\
            /torch.sum(solution(nodes_on_bc)**2, dim = 0)
          )
print(f"L2_error: {L2_error}")

end_time = time.time()     
training_time = end_time - start_time   
print("训练时间为：", training_time, "秒")

#画出误差图像

fig = plt.figure()
axes = fig.add_subplot()
qm = s.diff(solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[250, 250])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff')
fig.colorbar(qm)

plt.show()
