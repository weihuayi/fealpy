
import numpy as np

CONTEXT = 'numpy'

from fealpy.pde.helmholtz_2d import HelmholtzData2d

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer

from fealpy.np.mesh import TriangleMesh
from fealpy.np.functionspace import LagrangeFESpace
from fealpy.np.fem import (
    BilinearForm, LinearForm,
    ScalarMassIntegrator,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarRobinSourceIntegrator,
    ScalarRobinBoundaryIntegrator
)

from scipy.sparse.linalg import spsolve

from matplotlib import pyplot as plt 


k = 1
kappa = k * 1j
pde = HelmholtzData2d(k=k) 
domain = pde.domain()
NX, NY = 64, 64

tmr = timer()

mesh_plot = TMD.from_box(nx=NX, ny=NY)
next(tmr)

mesh = TriangleMesh.from_box(domain, nx=NX, ny=NY)

space = LagrangeFESpace(mesh, p=1)
space.ftype = np.complex128
tmr.send('mesh_and_space')

b = BilinearForm(space)
b.add_integrator([ScalarDiffusionIntegrator(), 
                  ScalarMassIntegrator(-k**2)])
b.add_integrator(ScalarRobinBoundaryIntegrator(kappa))

l = LinearForm(space)
l.add_integrator(ScalarSourceIntegrator(pde.source))
l.add_integrator(ScalarRobinSourceIntegrator(pde.robin))
tmr.send('forms')

A = b.assembly() 
F = l.assembly()
tmr.send('assembly')

uh = space.function(dtype=np.complex128)
uh[:] = spsolve(A, F)
value = space.value(uh, np.array([[1/3, 1/3, 1/3]], dtype=np.float64))
tmr.send('spsolve')
next(tmr)

fig = plt.figure(figsize=(12, 9))
fig.tight_layout()
fig.suptitle('Solving Helmholtz equation on 2D Triangle mesh')

axes = fig.add_subplot(1, 2, 1)
mesh_plot.add_plot(axes, cellcolor=np.real(value), cmap='jet', linewidths=0, showaxis=True)
axes = fig.add_subplot(1, 2, 2)
mesh_plot.add_plot(axes, cellcolor=np.imag(value), cmap='jet', linewidths=0, showaxis=True)
plt.show()
