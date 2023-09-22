import time

from matplotlib import pyplot as plt
from scipy.linalg import solve
import torch
import torch.nn as nn
from torch.special import bessel_j0

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import QuadrangleCollocator
from fealpy.mesh import TriangleMesh

#方程形式
"""

    \Delta u(x,y) + k^2 * u(x,y) = 0 ,                            (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{k^2/2} * x + \sqrt{k^2/2} * y) ,         (x,y)\in \partial\Omega

"""

#超参数(配置点个数、源点个数、波数)
num_of_col_bd = 5000
num_of_source = 5000
k = torch.tensor(1000, dtype=torch.float64)

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
    
source_nodes = QuadrangleCollocator(num_of_source, [[-2.5, 2.5],[-2.5, 2.5]]).run()
pikf_layer = PIKF_layer(source_nodes)
net_PIKFNN = nn.Sequential(
                           pikf_layer,
                           nn.Linear(num_of_source, 1, dtype=torch.float64, bias=False)
                           )
s = Solution(net_PIKFNN)

#真解及边界条件
def solution(p:torch.Tensor) -> torch.Tensor:

    x = p[...,0:1]
    y = p[...,1:2]
    return torch.sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)

def dirichletBC(p:torch.Tensor) -> torch.Tensor:
    return solution(p)

# 更新网络参数
start_time = time.time()

col_bd = QuadrangleCollocator(num_of_col_bd, [[-1, 1],[-1, 1]]).run()

A = pikf_layer.kernel_func(col_bd).detach().numpy()
b = dirichletBC(col_bd).detach().numpy()
alpha = solve(A, b)
net_PIKFNN[1].weight.data = torch.from_numpy(alpha).T
del alpha 

end_time = time.time()     
time_of_computation = end_time - start_time   
print("计算时间为：", time_of_computation, "秒")

#用网格计算误差
mesh_err = TriangleMesh.from_box([-1, 1, -1, 1], nx=30, ny=30)
error = s.estimate_error_tensor(solution, mesh_err) 
print(f"L-2 error: {error.item()}")  
print("Time:", time_of_computation, "s")

#可视化两者偏差
fig = plt.figure()
axes = fig.add_subplot()
qm = s.diff(solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[300, 300])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff')
fig.colorbar(qm)
plt.show()


