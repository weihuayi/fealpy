import sys
import numpy as np 
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.linear_elasticity_model import CantileverBeam2d
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

n = int(sys.argv[1])
pde = CantileverBeam2d()
mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=1)
uh = space.function(dim=2)
A = space.linear_elasticity_matrix(mu, lam)
F = space.source_vector(pde.source, dim=2)
bc = BoundaryCondition(space, dirichlet=pde.dirichlet,  neuman=pde.neuman)
bc.apply_neuman_bc(F)
A, F = bc.apply_dirichlet_bc(A, F, uh, 
    is_dirichlet_boundary=pde.is_dirichlet_boundary)

uh.T.flat[:] = spsolve(A, F)

error = space.integralalg.L2_error(pde.displacement, uh)
print(error)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
