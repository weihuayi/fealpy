import time

import torch
import torch.nn as nn
from torch.special import bessel_j0
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from matplotlib import pyplot as plt

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import QuadrangleCollocator
from fealpy.mesh import TriangleMesh

#方程形式
"""

    \Delta u(x,y) + k^2 * u(x,y) = 0 ,                                   (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{0.5 * k^2} * x + \sqrt{0.5 * k^2} * y) ,         (x,y)\in \partial\Omega

"""

#超参数(配置点个数、源点个数、波数)
NN = 1500
NS = 1500
lr = 0.1
iteration = 1000
k = torch.tensor(10, dtype=torch.float64)

#PIKF层
class PIKF_layer(nn.Module):
    def __init__(self, source_nodes: torch.Tensor) -> None:
        super().__init__()
        self.source_nodes = source_nodes

    def kernel_func(self, input: torch.Tensor) -> torch.Tensor:
        
        a = input[:, None, :] - self.source_nodes[None, :, :]
        r = torch.sqrt((a[..., 0:1]**2 + a[..., 1:2]**2).view(input.shape[0], -1))
        val = bessel_j0(k* r)/(2 * torch.pi)
        return val

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.kernel_func(p)

#源点、边界配置点采样
source_nodes = QuadrangleCollocator([[-2.5, 2.5], [-2.5, 2.5]]).run(NS)
col_bd = QuadrangleCollocator([[-1, 1], [-1, 1]]).run(NN)

#实例化
pikf_layer = PIKF_layer(source_nodes)
net_PIKFNN = nn.Sequential(
    pikf_layer,
    nn.Linear(NS, 1, dtype=torch.float64, bias=False)
)
s = Solution(net_PIKFNN)

#选择优化器和损失函数
optim = Adam(s.parameters(), lr=lr, betas=(0.9, 0.99))
mse_cost_func = nn.MSELoss(reduction='mean')
scheduler = StepLR(optim, step_size=50, gamma=0.85)

#真解及边界条件
def exact_solution(p:torch.Tensor) -> torch.Tensor:

    x = p[...,0:1]
    y = p[...,1:2]
    return torch.sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)

def bc(p:torch.Tensor) -> torch.Tensor:
    return s(p) - exact_solution(p)

#训练网络参数
start_time = time.time()
Loss = []
Error= []
mesh_err = TriangleMesh.from_box([-1, 1, -1, 1], nx=100, ny=100)
for epoch in range(iteration+1):
    optim.zero_grad()
    loss = mse_cost_func(bc(col_bd), torch.zeros_like(bc(col_bd))) 
    loss.backward(retain_graph=True)
    optim.step()
    scheduler.step()

    if epoch % 10 == 0:

        error = s.estimate_error_tensor(exact_solution, mesh_err)
        Error.append(error.detach().numpy())
        Loss.append(loss.detach().numpy())
        print(f"Epoch: {epoch}, Loss: {loss}")
        print(f"Error:{error.item()}")
        print('\n')

end_time = time.time()
time_of_computation = end_time - start_time
print("计算时间为：", time_of_computation, "秒")

#训练过程可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
y= range(1, 10*len(Error) +1,10)
ax1.plot(y, Error)
ax1.set_ylim(0, 0.2) 

loss = range(1, 10 * len(Loss) + 1, 10)
ax2.plot(loss, Loss)
ax2.set_ylim(0, 0.2) 

#可视化数值解、真解以及两者偏差
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 第一个子图
qm_1 = Solution(exact_solution).add_pcolor(axes[0], box=[-1, 1, -1, 1], nums=[150, 150])
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('u')
fig.colorbar(qm_1, ax=axes[0])

# 第二个子图
qm_2 = s.add_pcolor(axes[1], box=[-1, 1, -1, 1], nums=[150, 150])
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('u_PIKFNN')
fig.colorbar(qm_2, ax=axes[1])

# 第三个子图
qm_3 = s.diff(exact_solution).add_pcolor(axes[2], box=[-1, 1, -1, 1], nums=[150, 150])
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('diff')
fig.colorbar(qm_3, ax=axes[2])

plt.show()

