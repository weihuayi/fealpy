import numpy as np
import matplotlib.pyplot as plt
import sys

from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.pde.poisson_2d import CosCosData


n = int(sys.argv[1])
p = int(sys.argv[2])
pde = CosCosData()

mesh = pde.init_mesh(n=n, meshtype='quadtree')

pmesh = mesh.to_pmesh()

space = ConformingVirtualElementSpace2d(pmesh, p=p)
uI = space.interpolation(pde.solution)
S = space.project_to_smspace(uI)

error = space.integralalg.L2_error(pde.solution, S.value)
print(error)
uh = space.grad_recovery(uI)
S = space.project_to_smspace(uh)

def f(x, cellidx):
    val = S.value(x, cellidx)
    return np.sum(val**2, axis=-1)

a = space.integralalg.integral(f, celltype=True)
a = np.sqrt(a)
print(a)

gu = pde.gradient


error = space.integralalg.L2_error(gu, S.value)
print(error)

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
plt.show()


