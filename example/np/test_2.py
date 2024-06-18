
import numpy as np

CONTEXT = 'torch'

from numpy.typing import NDArray

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer

from fealpy.np.mesh import TriangleMesh
from fealpy.np.functionspace import LagrangeFESpace
from fealpy.np.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from scipy.sparse.linalg import spsolve

from numpy import cos, pi, tensordot

from matplotlib import pyplot as plt


NX, NY = 64, 64

def source(points: NDArray):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype}
    coef = np.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return np.einsum(
        "b, b... -> b...",
        2*coef**2,
        cos(tensordot(coef, x, axes=0)) * cos(tensordot(coef, y, axes=0))
    )


def solution(points: NDArray):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype}
    coef = np.linspace(pi/2, 5*pi, 10).to(**kwargs)
    return cos(tensordot(coef, x, axes=0)) * cos(tensordot(coef, y, axes=0))


tmr = timer()

mesh_numpy = TMD.from_box(nx=NX, ny=NY)
next(tmr)
mesh = TriangleMesh.from_box(nx=NX, ny=NY)
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
uh = spsolve(A, F, maxiter=5000, batch_first=True)
value = space.value(uh, np.array([[1/3, 1/3, 1/3]], dtype=np.float64)).squeeze(0)
value = value.cpu().numpy()
tmr.send('spsolve')
next(tmr)

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Parallel solving Poisson equation on 2D Triangle mesh')

for i in range(10):
    axes = fig.add_subplot(3, 4, i+1)
    mesh_numpy.add_plot(axes, cellcolor=value[i, :], cmap='jet', linewidths=0, showaxis=True)

plt.show()
