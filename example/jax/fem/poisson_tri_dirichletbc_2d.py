
import numpy as np

CONTEXT = 'jax'

import jax.numpy as jnp

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer
from fealpy.jax.utils import Array

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from fealpy.jax.mesh import TriangleMesh

from fealpy.jax.functionspace import LagrangeFESpace
from fealpy.jax.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
# from fealpy.np.fem import DirichletBC
from scipy.sparse.linalg import spsolve

from jax.numpy import cos, pi

from matplotlib import pyplot as plt

NX, NY = 64, 64
def source(points: Array):
    x = points[..., 0]
    y = points[..., 1]
    coef = 2 * pi
    return 2 * coef**2 * cos(coef * x) * cos(coef * y)

def solution(points: Array):
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
values = A.data
indices = A.indices

# 将 JAX 稀疏矩阵的索引转换为 SciPy CSR 矩阵所需的行和列索引
rows = indices[:, 0]
cols = indices[:, 1]

# # 创建 SciPy 的 CSR 矩阵
A = csr_matrix((values, (rows, cols)), shape=A.shape)
A, F = DirichletBC(space, gD = solution).apply(A, F, uh)
tmr.send('dirichlet')

uh = spsolve(A, F)
value = space.value(uh, jnp.array([[1/3, 1/3, 1/3]], dtype=jnp.float64))
tmr.send('spsolve')
next(tmr)

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Solving Poisson equation on 2D Triangle mesh')

axes = fig.add_subplot(1, 1, 1)
mesh_numpy.add_plot(axes, cellcolor=np.array(value), cmap='jet', linewidths=0, showaxis=True)

plt.show()
