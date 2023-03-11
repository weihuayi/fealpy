from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Linear, Tanh
from torch.autograd import Variable

from fealpy.pinn.machine import Solution, LearningMachine
from fealpy.pinn.sampler import ISampler
from fealpy.pinn.grad import grad_by_fts

Function = Callable[[torch.Tensor], torch.Tensor]


### Load Feature Line Model

print("载入特征线模型...")
state_dict = torch.load("FeatureLines.pth")

class InitialValue(Solution):
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        t = p[:, 0:1]
        xy = p[:, 1:3]
        return self.net(p) - self.net(torch.cat([torch.zeros_like(t), xy], dim=1)) + xy

pinn = Sequential(
    Linear(3, 32),
    Tanh(),
    Linear(32, 16),
    Tanh(),
    Linear(16, 8),
    Tanh(),
    Linear(8, 4),
    Tanh(),
    Linear(4, 2)
)

fl = InitialValue(pinn)
fl.load_state_dict(state_dict)
fl.training = False


### Define Levelset Model

print("训练水平集...")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_l1 = nn.Linear(3, 64)
        self.hidden_l2 = nn.Linear(64, 32)
        self.hidden_l3 = nn.Linear(32, 16)
        self.hidden_l4 = nn.Linear(16, 8)
        self.hidden_l5 = nn.Linear(8, 4)
        self.hidden_l6 = nn.Linear(4, 2)
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x):
        out = torch.tanh(self.hidden_l1(x))
        out = torch.tanh(self.hidden_l2(out))
        out = torch.tanh(self.hidden_l3(out))
        out = torch.tanh(self.hidden_l4(out))
        out = torch.tanh(self.hidden_l5(out))
        out = torch.tanh(self.hidden_l6(out))
        return self.output_layer(out)


def circle(p: torch.Tensor):
    x = p[..., 0:1]
    y = p[..., 1:2]
    val = torch.sqrt((x-0.5)**2+(y-0.75)**2) - 0.15
    return val


class InitialValue2(Solution):
    def forward(self, p: torch.Tensor):
        t = p[..., 0:1]
        x = p[..., 1:3]
        return self.net(p) + circle(x) - self.net(torch.cat([torch.zeros_like(t), x], dim=-1))


phi = InitialValue2(Net())
lm = LearningMachine(phi)
domain = [0, 1, 0, 1]

mse_cost_function = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(phi.parameters(), lr=0.0005)
sampler1 = ISampler(10000, [[0, 0.5], [0, 1], [0, 1]], requires_grad=True)
sampler2 = ISampler(10000, [[0.125, 0.375], [0, 1], [0, 1]], requires_grad=True)


def velocity_field(p: torch.Tensor):
    x = p[..., 0]
    y = p[..., 1]
    u = torch.zeros(p.shape)
    u[..., 0] = torch.sin((np.pi*x))**2 * torch.sin(2*np.pi*y)
    u[..., 1] = -torch.sin((np.pi*y))**2 * torch.sin(2*np.pi*x)
    return u


def levelset_equation(tx: torch.Tensor, phi):
    phi_val = phi(tx)
    grad_phi = grad_by_fts(phi_val, tx, create_graph=True)
    phi_t = grad_phi[:, 0:1]
    phi_x = grad_phi[:, 1:3]
    u_phi_x = torch.einsum("ix, ix -> i", phi_x, velocity_field(tx[..., 1:3])).reshape(-1, 1)
    return phi_t + u_phi_x


iterations = 100

for epoch in range(iterations):
    optimizer.zero_grad()

    # loss of pde
    mse_f = lm.loss(sampler1, levelset_equation)

    pts = sampler2.run()
    phi_c = phi(pts)
    m = pts.shape[0]

    time_c = pts[:, 0:1]
    time_delta = 0.125*torch.ones((m, 1))

    pts_p = fl(torch.cat([-time_delta, pts[:, 1:3]], dim=1))
    phi_p = phi(torch.cat([time_c - time_delta, pts_p], dim=1))
    pts_f = fl(torch.cat([time_delta, pts[:, 1:3]], dim=1))
    phi_f = phi(torch.cat([time_c + time_delta, pts_p], dim=1))

    mse_fl = mse_cost_function(phi_c, phi_p) + mse_cost_function(phi_c, phi_f)

    # backward
    loss = 0.8*mse_f + 0.2*mse_fl
    loss.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        if (epoch) % 500 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.data}")


### Estimate error

print("FEM 求解...")
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian
from fealpy.solver import LevelSetFEMFastSolver


@cartesian
def velocity_field2(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

@cartesian
def circle2(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val


domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=100, ny=100, meshtype='tri')

timeline = UniformTimeLine(0, 0.5, 100)
dt = timeline.dt

space = LagrangeFiniteElementSpace(mesh, p=1)
phi0 = space.interpolation(circle2)
u = space.interpolation(velocity_field2, dim=2)


M = space.mass_matrix()
C = space.convection_matrix(c = u).T
A = M + dt/2*C

diff = []
measure = space.function()

solver = LevelSetFEMFastSolver(A)

for i in range(1000):

    t1 = timeline.next_time_level()
    print("t1=", t1)

    #计算面积
    measure[phi0 > 0] = 0
    measure[phi0 <=0] = 1
    diff.append(abs(space.integralalg.integral(measure) - (np.pi)*0.15**2))

    b = M@phi0 - dt/2*(C@phi0)

    phi0[:] = solver.solve(b, tol=1e-12)

    # 时间步进一层
    timeline.advance()

final_phi = phi.fixed([0, ], [0.5, ])
error = final_phi.estimate_error(phi0, squeeze=True)

print(f"计算误差：{error}")

### Draw the solution

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors

t = [0, 0.125, 0.25, 0.375, 0.5]
x = np.linspace(domain[0], domain[1], 100)
y = np.linspace(domain[2], domain[3], 100)
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False)

fig = plt.figure()
norm = colors.BoundaryNorm(np.array([-1, 0, 1]), ncolors=256)

for i, tp in enumerate(t):
    points = torch.cat([tp*torch.ones_like(pt_x), pt_x, pt_y], dim=1)
    phi_val = phi(points).data.cpu().numpy().reshape(100, 100)
    ax = fig.add_subplot(2, 3, i+1)
    pcm = ax.pcolormesh(ms_x, ms_y, phi_val, cmap=cm.RdYlBu_r)
    ax.contour(ms_x, ms_y, phi_val, [-0.5, 0, 0.5])
    ax.set_title(f't = {tp}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax, extend='both')

plt.show()
