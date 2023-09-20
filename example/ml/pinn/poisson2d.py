"""
PINN for 2-d poisson equation
"""
from time import time

import torch
import torch.nn as nn
from torch import Tensor, cos, float64
from torch.optim.lr_scheduler import ExponentialLR

from fealpy.mesh import TriangleMesh, UniformMesh2d
from fealpy.ml.grad import gradient
from fealpy.ml.modules import Solution

NEW_MODEL = True
PI = torch.pi

def exact_solution(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return cos(PI * x) * cos(PI * y)

def pde_part(p: Tensor, phi):
    x = p[:, 0:1]
    y = p[:, 1:2]
    u = phi(p)
    u_x, u_y = gradient(u, p, create_graph=True, split=True)
    u_xx, _ = gradient(u_x, p, create_graph=True, split=True)
    _, u_yy = gradient(u_y, p, create_graph=True, split=True)
    return u_xx + u_yy + 2 * PI**2 * cos(PI * x) * cos(PI * y)

def bc_part(p: Tensor, phi):
    return phi(p) - exact_solution(p)


class Cos(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.cos(input)


EXTC = 100
HC = 1/EXTC

start_time = time()
NN: int = 64
pinn = nn.Sequential(
    nn.Linear(2, NN, dtype=float64),
    nn.Tanh(),
    nn.Linear(NN, NN//2, dtype=float64),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4, dtype=float64),
    Cos(),
    nn.Linear(NN//4, 1, dtype=float64)
)

if not NEW_MODEL:
    try:
        state_dict = torch.load("pinn_poisson2d.pth")
        pinn.load_state_dict(state_dict)
    except:
        pass


optim = torch.optim.Adam(pinn.parameters(), lr=0.01, weight_decay=0)
lrs = ExponentialLR(optim, gamma=0.9977)
loss_fn = nn.MSELoss()
s = Solution(pinn)


mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))
col_in.requires_grad_(True)
col_bd.requires_grad_(False)

mesh_err = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)

for epoch in range(2000):
    optim.zero_grad()
    out_pde = pde_part(col_in, pinn)
    out_bc = bc_part(col_bd, pinn)
    mse_f = loss_fn(out_pde, torch.zeros_like(out_pde))
    mse_b = loss_fn(out_bc, torch.zeros_like(out_bc))
    loss = 0.1*mse_f + 0.9*mse_b
    loss.backward()
    optim.step()
    lrs.step()

    with torch.autograd.no_grad():
        if epoch % 100 == 99:
            print(f"Epoch: {epoch+1} | Loss: {loss.data}")

end_time = time()

### Estimate error

error = s.estimate_error_tensor(exact_solution, mesh_err)

print(f"L-2 error: {error.item()}")
print(f"Time: {end_time - start_time}")

state_dict = pinn.state_dict()
torch.save(state_dict, "pinn_poisson2d.pth")

### Draw the result

import matplotlib.pyplot as plt
fig = plt.figure("PINN for 2d poisson equation")

axes = fig.add_subplot(111)
qm = s.diff(exact_solution).add_pcolor(axes, [0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('t')
axes.set_ylabel('x')
fig.colorbar(qm)

plt.show()
