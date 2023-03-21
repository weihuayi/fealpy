
import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Linear, Tanh

from fealpy.pinn.machine import Solution, LearningMachine
from fealpy.pinn.sampler import ISampler
from fealpy.pinn.grad import grad_by_fts
from fealpy.pinn.hyperparams import AutoTest

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace.Function import Function


### Load Feature Line Model

print("载入特征线模型...")
state_dict = torch.load("FeatureLines.pth")

class InitialValue(Solution):
    def forward(self, p: torch.Tensor) -> torch.Tensor:
        t = p[:, 0:1]
        xy = p[:, 1:3]
        return self.net(p) - self.net(torch.cat([torch.zeros_like(t), xy], dim=1)) + xy

pinn = Sequential(
    Linear(3, 64),
    Tanh(),
    Linear(64, 32),
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

sets = {
    "tau": [0.001, 0.003, 0.01, 0.03, 0.1, 0.25],
    "nfl": [1000, 10000],
}

at = AutoTest(sets)
at.set_autosave('levelset2d.json', mode='replace')

lr = 0.0005
Nf = 10000

for TAU, Nfl in at.run():

    ws = [1.0, 1/TAU**2]

    pinn2 = Sequential(
        Linear(3, 64),
        Tanh(),
        Linear(64, 32),
        Tanh(),
        Linear(32, 16),
        Tanh(),
        Linear(16, 8),
        Tanh(),
        Linear(8, 4),
        Tanh(),
        Linear(4, 2),
        Tanh(),
        Linear(2, 1)
    )


    phi = InitialValue2(pinn2)
    lm = LearningMachine(phi)
    domain = [0, 1, 0, 1]
    T = 1.0

    mse_cost_function = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(phi.parameters(), lr=lr)
    sampler1 = ISampler(Nf, [[0, 1], [0, 1], [0, 1]], requires_grad=True)
    sampler2 = ISampler(Nfl, [[TAU, T-TAU], [0, 1], [0, 1]], requires_grad=True)


    iterations = 5000

    for epoch in range(iterations):
        optimizer.zero_grad()

        # loss of pde
        mse_f = lm.loss(sampler1, levelset_equation)

        pts = sampler2.run()
        m = pts.shape[0]
        delta = torch.ones((m, 1)) * TAU
        phi_c = phi(pts)

        pts_f = fl(torch.cat([delta, pts[:, 1:3]], dim=1))
        phi_f = phi(torch.cat([pts[:, 0:1] + delta, pts_f], dim=1))

        pts_p = fl(torch.cat([-delta, pts[:, 1:3]], dim=1))
        phi_p = phi(torch.cat([pts[:, 0:1] - delta, pts_p], dim=1))

        mse_fl = mse_cost_function(phi_c, phi_f) + mse_cost_function(phi_c, phi_p)

        # backward
        loss = ws[0]*mse_f + ws[1]*mse_fl
        loss.backward()
        optimizer.step()

        with torch.autograd.no_grad():
            if (epoch) % 500 == 0:
                print(f"Epoch: {epoch} | Loss: {loss.data}")

    ### Estimate error

    mesh = MF.boxmesh2d(domain, nx=100, ny=100, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=5)
    phi0 = Function(space, array=np.load('LevelSet2d_5.npy'))

    final_phi = phi.fixed([0, ], [T, ])
    error = final_phi.estimate_error(phi0, squeeze=True)

    print(f"计算误差：{error}")
    at.record(time=at.item_runtime(), loss=float(loss), error=error)
