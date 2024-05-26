
import torch
from torch import Tensor

CONTEXT = 'torch'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarBoundarySourceIntegrator
)
from fealpy.torch.solver import sparse_cg

from torch import cos, pi, tensordot

from fealpy.ml import timer
from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NX, NY = 64, 64

def source(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    coef = torch.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return torch.einsum(
        "b, ...b -> ...b",
        2*coef**2,
        cos(tensordot(x, coef, dims=0)) * cos(tensordot(y, coef, dims=0))
    )


def neumann(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    theta = torch.arctan2(y, x)
    freq = torch.arange(1, 11, **kwargs)
    return cos(tensordot(theta, freq, dims=0))


tmr = timer()

mesh_numpy = TMD.from_box(nx=NX, ny=NY)
cell = mesh_numpy.ds.cell
node = mesh_numpy.node
next(tmr)
mesh = TriangleMesh(
    torch.from_numpy(node).to(device),
    torch.from_numpy(cell).to(device),
)
NC = cell.shape[0]


space = LagrangeFESpace(mesh, p=3)
tmr.send('mesh_and_space')

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator())

lform = LinearForm(space, batch_size=10)
# lform.add_integrator(ScalarSourceIntegrator(source, batched=True))
lform.add_integrator(ScalarBoundarySourceIntegrator(neumann, batched=True))
tmr.send('forms')

A = bform.assembly()
F = lform.assembly()

uh = torch.zeros((space.number_of_global_dofs(), 10), dtype=torch.float64, device=device)
tmr.send('assembly')


A = A.to_sparse_csr()
uh = sparse_cg(A, F, uh, maxiter=5000)
uh = uh.detach()
value = space.value(uh, torch.tensor([[1/3, 1/3, 1/3]], device=device, dtype=torch.float64)).squeeze(0)
value = value.cpu().numpy()
print(value.shape)
# uh = uh.cpu().numpy()
tmr.send('solve(cg)')
tmr.send('stop')

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Parallel solving Poisson equation on 2D Triangle mesh')

for i in range(10):
    axes = fig.add_subplot(3, 4, i+1)
    mesh_numpy.add_plot(axes, cellcolor=value[:, i], cmap='jet', linewidths=0, showaxis=True)

plt.show()
