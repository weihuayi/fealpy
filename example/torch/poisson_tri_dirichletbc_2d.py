
import torch
from torch import Tensor

CONTEXT = 'torch'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer

from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from fealpy.torch.solver import sparse_cg

from torch import cos, pi, tensordot

from matplotlib import pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NX, NY = 64, 64

def source(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    coef = torch.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return torch.einsum(
        "b, b... -> b...",
        2*coef**2,
        cos(tensordot(coef, x, dims=0)) * cos(tensordot(coef, y, dims=0))
    )


def solution(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    coef = torch.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return cos(tensordot(coef, x, dims=0)) * cos(tensordot(coef, y, dims=0))


tmr = timer()

mesh_numpy = TMD.from_box(nx=NX, ny=NY)
next(tmr)
mesh = TriangleMesh.from_box(nx=NX, ny=NY, device=device)
NC = mesh.number_of_cells()

space = LagrangeFESpace(mesh, p=3)
tmr.send('mesh_and_space')

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator())

lform = LinearForm(space, batch_size=10)
lform.add_integrator(ScalarSourceIntegrator(source, batched=True))
tmr.send('forms')

A = bform.assembly()
F = lform.assembly()
tmr.send('assembly')

A, F = DirichletBC(space).apply(A, F, gd=solution)
tmr.send('dirichlet')

A = A.to_sparse_csr()
uh = sparse_cg(A, F, maxiter=5000, batch_first=True)
uh = uh.detach()
value = space.value(uh, torch.tensor([[1/3, 1/3, 1/3]], device=device, dtype=torch.float64)).squeeze(0)
value = value.cpu().numpy()
tmr.send('solve(cg)')
next(tmr)

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Parallel solving Poisson equation on 2D Triangle mesh')

for i in range(10):
    axes = fig.add_subplot(3, 4, i+1)
    mesh_numpy.add_plot(axes, cellcolor=value[i, :], cmap='jet', linewidths=0, showaxis=True)

plt.show()
