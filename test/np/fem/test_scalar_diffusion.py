
import numpy as np

CONTEXT = 'numpy'

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

from numpy import cos, pi

from matplotlib import pyplot as plt

NX, NY = 64, 64
def source(points: NDArray):
    x = points[..., 0]
    y = points[..., 1]
    coef = 2 * pi
    return 2 * coef**2 * cos(coef * x) * cos(coef * y)

def solution(points: NDArray):
    x = points[..., 0]
    y = points[..., 1]
    coef = 2 * pi
    return cos(coef * x) * cos(coef * y)

tmr = timer()

mesh_numpy = TMD.from_box(nx=NX, ny=NY)
next(tmr)

mesh = TriangleMesh.from_box(nx=NX, ny=NY)
NC = mesh.number_of_cells()

space = LagrangeFESpace(mesh, p=3)
ldof = space.number_of_local_dofs()
print('ldof:', ldof)
tmr.send('mesh_and_space')

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator())

lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(source))

tmr.send('forms')

A = bform.assembly()
F = lform.assembly()
tmr.send('assembly')

uh = space.function()
A, F = DirichletBC(space, gD = solution).apply(A, F, uh)
tmr.send('dirichlet')

uh = spsolve(A, F)
value = space.value(uh, np.array([[1/3, 1/3, 1/3]], dtype=np.float64))
tmr.send('spsolve')
next(tmr)

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Solving Poisson equation on 2D Triangle mesh')

axes = fig.add_subplot(1, 1, 1)
mesh_numpy.add_plot(axes, cellcolor=value, cmap='jet', linewidths=0, showaxis=True)

plt.show()