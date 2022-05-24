
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData

from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.boundarycondition import DirichletBC


pde = CosCosData()

mesh = pde.init_mesh(n=5, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=1)

uh = space.function() # (NN, )

A = space.stiff_matrix()
F = space.source_vector(pde.source)

bc = DirichletBC(space, pde.dirichlet)

A, F = bc.apply(A, F, uh)

uh[:] = spsolve(A, F)

L2error = space.integralalg.error(pde.solution, uh) # L_2 error
H1error = space.integralalg.error(pde.gradient, uh.grad_value)

print(L2error)
print(H1error)


uh.add_plot(plt, cmap='rainbow')
plt.show()


