
import torch
from torch import Tensor, float64, sin, exp
import torch.nn as nn
from torch.optim import Adam

from fealpy.ml.modules import BoxTimeDBCSolution2d, Solution
from fealpy.ml.grad import gradient
from fealpy.ml.sampler import ISampler

PI = torch.pi

# 定义神经网络
pinn = nn.Sequential(
    nn.Linear(2, 32, dtype=float64),
    nn.Tanh(),
    nn.Linear(32, 16, dtype=float64),
    nn.Tanh(),
    nn.Linear(16, 8, dtype=float64),
    nn.Tanh(),
    nn.Linear(8, 4, dtype=float64),
    nn.Tanh(),
    nn.Linear(4, 1, dtype=float64)
)

# 定义优化器、采样器、训练器
optimizer = Adam(pinn.parameters(), lr=0.001)
mse_fn = nn.MSELoss()
sampler = ISampler([[0, 1], [0, 2]], requires_grad=True)
s = BoxTimeDBCSolution2d(pinn)

# 设置初边值条件
@s.set_ic
def inital(p: Tensor):
    _, x = torch.split(p, 1, dim=-1)
    return sin(PI * x)

s.set_box([0.0, 2.0])

@s.set_bc
def boundary(p: Tensor):
    t, _ = torch.split(p, 1, dim=-1)
    return torch.zeros_like(t)

# 定义 PDE
def heat_equation(p: torch.Tensor, u):
    phi = u(p)
    phi_t, phi_x = gradient(phi, p, create_graph=True, split=True)
    _, phi_xx = gradient(phi_x, p, create_graph=True, split=True)
    return phi_t - 0.1*phi_xx

# 开始训练
for epoch in range(1, 1001):
    optimizer.zero_grad()
    pts = sampler.run(1000)
    output = heat_equation(pts, s)
    loss = mse_fn(output, torch.zeros_like(output))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        with torch.no_grad():
            print(f"Epoch: {epoch}| Loss: {loss.item()}")


### Draw the result

def exact_solution(p: Tensor):
    t, x = torch.split(p, 1, dim=-1)
    return exp(-0.1 * PI**2 * t) * sin(PI * x)


from matplotlib import pyplot as plt

fig = plt.figure("PINN(with TFC) for the heat equation")
axes = fig.add_subplot(121, projection='3d')
s.add_surface(axes, box=[0, 1, 0, 2], nums=[100, 200])
axes.set_title('solution')
axes.set_xlabel('t')
axes.set_ylabel('x')
axes.set_zlabel('phi')

axes = fig.add_subplot(122)
qm = s.diff(exact_solution).add_pcolor(axes, box=[0, 1, 0, 2], nums=[100, 200])
fig.colorbar(qm)
axes.set_title('bias')
axes.set_xlabel('t')
axes.set_ylabel('x')

plt.show()
