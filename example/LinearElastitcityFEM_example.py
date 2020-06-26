
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.linear_elasticity_model import Hole2d, Model2d

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition
from fealpy.tools.show import showmultirate


m = int(sys.argv[1])
p = int(sys.argv[2])
maxit = int(sys.argv[3])

if m == 1:
    pde = Model2d()
if m == 2:
    pde = Hole2d(lam=10, mu=1.0)

errorType = ['$||u - u_h||_{0}$']
Ndof = np.zeros((maxit,))
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

mesh = pde.init_mesh()

mesh.uniform_refine(4)
space = LagrangeFiniteElementSpace(mesh, p=p)

GD = space.geo_dimension()
bc = BoundaryCondition(space, dirichlet=pde.dirichlet)

A = space.linear_elasticity_matrix(pde.mu, pde.lam)
F = space.source_vector(pde.source, dim=GD)

for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p=p)
    Ndof[i] = space.number_of_global_dofs()
    GD = space.geo_dimension()
    bc = BoundaryCondition(space, dirichlet=pde.dirichlet)


    uh = space.function(dim=GD)
    
    bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
    A, F = bc.apply_dirichlet_bc(A, F, uh)

    uh.T.flat[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.displacement, uh)

    mesh.uniform_refine()

print(errorMatrix)
showmultirate(plt, 1, Ndof, errorMatrix, errorType)
plt.show()
