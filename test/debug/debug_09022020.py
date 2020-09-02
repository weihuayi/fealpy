
import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.pde.poisson_2d import CosCosData

from fealpy.solver import HighOrderLagrangeFEMFastSolver


p = int(sys.argv[1])
n = int(sys.argv[2])

box = [0, 1, 0, 1]
mf = MeshFactory()
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='tri')

pde = CosCosData()

space = LagrangeFiniteElementSpace(mesh, p=p)

uh = space.function()
isBdDof = space.set_dirichlet_bc(uh, pde.dirichlet)

A = space.stiff_matrix()
F = space.source_vector(pde.source)
P = mesh.linear_stiff_matrix()
I = space.linear_interpolation_matrix()

solver = HighOrderLagrangeFEMFastSolver(A, F, P, I, isBdDof)

uh = solver.solve(uh, F)

error = space.integralalg.error(pde.solution, uh)
print(error)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()



