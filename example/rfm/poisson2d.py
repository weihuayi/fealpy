"""
Solve the poisson 2d problem using global random feature functions.
"""


import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from fealpy.pde.poisson_2d import CosCosData
from fealpy.pinn.modules import RandomFeatureFlat, RandomFeature
from fealpy.pinn.sampler import ISampler, BoxBoundarySampler, get_mesh_sampler
from fealpy.pinn.grad import gradient
from fealpy.mesh import UniformMesh2d


class CosCosTorch(CosCosData):
    def dirichlet(self, p):
        x = p[..., 0:1]
        y = p[..., 1:2]
        pi = torch.pi
        val = torch.cos(pi*x)*torch.cos(pi*y)
        return val

pde = CosCosTorch()


def pde_part(p: torch.Tensor, u):
    u_x, u_y = gradient(u, p, create_graph=True, split=True)
    u_xx, _ = gradient(u_x, p, create_graph=True, split=True)
    _, u_yy = gradient(u_y, p, create_graph=True, split=True)

    return u_xx + u_yy + np.pi**2 * u

def bc(x: torch.Tensor, u):
    return u - pde.dirichlet(x).unsqueeze(-1)


mesh = UniformMesh2d((0, 2, 0, 2), h=(0.5, 0.5), origin=(0.0, 0.0))
node = torch.from_numpy(mesh.entity('node'))

model = RandomFeatureFlat(50, 4, centers=node, radius=0.25, in_dim=2,
                          activate=torch.cos, print_status=True)
sampler = get_mesh_sampler(40, mesh, requires_grad=True)
# sampler = ISampler(1000, [[0, 1], [0, 1]], requires_grad=True)
sampler_bc = BoxBoundarySampler(1000, [0.0, 0.0], [1.0, 1.0], requires_grad=True)
optim = Adam(model.um, 1e-3)
loss_fn = MSELoss(reduction='mean')


MAX_ITER = 100

for epoch in range(MAX_ITER):
    optim.zero_grad()

    # s = sampler.run()
    # out = model(s)
    # pde_out = pde_part(s, out)
    # loss_pde = loss_fn(pde_out, torch.zeros_like(pde_out))

    s = sampler_bc.run()
    out = model(s)
    bc_out = bc(s, out)
    loss_bc = loss_fn(bc_out, torch.zeros_like(bc_out))

    loss = 0.95*loss_bc

    loss.backward()
    optim.step()

    if epoch % 20 == 19:
        with torch.no_grad():
            print(f"Epoch: {epoch+1}| Loss: {loss.data}")


from matplotlib import pyplot as plt
from matplotlib import cm

x = np.linspace(0, 1, 90, dtype=np.float64)
y = np.linspace(0, 1, 90, dtype=np.float64)

data, (mx, my) = model.meshgrid_mapping(x, y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(mx, my, data)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()
