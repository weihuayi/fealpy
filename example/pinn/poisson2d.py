from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import MeshFactory as Mf
from fealpy.pinn import LearningMachine, gradient, Solution
from fealpy.pinn.sampler import BoxBoundarySampler, TriangleMeshSampler

import torch
import torch.nn as nn
import numpy as np

class RecCosh(nn.Module):
    r"""Uses the element-wise activation function.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 1/torch.cosh(input)

pde = CosCosData()

NN: int = 64
pinn = nn.Sequential(
    nn.Linear(2, NN),
    RecCosh(),
    nn.Linear(NN, NN//2),
    RecCosh(),
    nn.Linear(NN//2, NN//4),
    RecCosh(),
    nn.Linear(NN//4, 1)
)

def pde_part(p: torch.Tensor, phi):
    u = phi(p)
    u_x, u_y = gradient(u, p, create_graph=True, split=True)
    u_xx, _ = gradient(u_x, p, create_graph=True, split=True)
    _, u_yy = gradient(u_y, p, create_graph=True, split=True)

    return u_xx + u_yy + np.pi**2 * u

def bc(x: torch.Tensor, phi):
    return phi(x) - pde.dirichlet(x).unsqueeze(-1)


mesh = Mf.boxmesh2d([0, 1, 0, 1], nx=10, ny=10)
optim = torch.optim.Adam(pinn.parameters(), lr=0.01, weight_decay=0)
s = Solution(pinn)
lm = LearningMachine(s)
# sampler1 = ISampler(300, [[0, 1], [0, 1]], requires_grad=True)
sampler1 = TriangleMeshSampler(5, mesh, requires_grad=True)
sampler2 = BoxBoundarySampler(3000, [0, 0], [1, 1])


for epoch in range(1200):
    optim.zero_grad()
    mse_f = lm.loss(sampler1, pde_part)
    mse_b = lm.loss(sampler2, bc)
    loss = 0.1*mse_f + 0.9*mse_b
    loss.backward()
    optim.step()

    with torch.autograd.no_grad():
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.data}")


### Estimate error

from fealpy.functionspace import LagrangeFiniteElementSpace

mesh2 = Mf.boxmesh2d([0, 1, 0, 1], nx=10, ny=10)
space = LagrangeFiniteElementSpace(mesh2)
error = s.estimate_error(pde.solution, mesh2, squeeze=True)

print(f"Error: {np.sqrt(error)}")

### Draw the result

from matplotlib import cm
import matplotlib.pyplot as plt
fig = plt.figure()

x = np.linspace(0, 1, 30)
y = np.linspace(0, 1, 30)
u, (X, Y) = s.meshgrid_mapping(x, y)

axes = fig.add_subplot(1, 1, 1, projection='3d')
axes.plot_surface(X, Y, u, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
axes.set_xlabel('t')
axes.set_ylabel('x')
axes.set_zlabel('u')
plt.show()
