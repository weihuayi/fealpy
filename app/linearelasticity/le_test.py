import sys
import numpy as np 
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.linear_elasticity_model import CantileverBeam2d
from fealpy.pde.linear_elasticity_model import HuangModel2d
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

n = int(sys.argv[1])
p = int(sys.argv[2])
scale = float(sys.argv[3])
pde = CantileverBeam2d()
#pde = HuangModel2d()
mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)

space = LagrangeFiniteElementSpace(mesh, p=p)
uh = space.function(dim=2)
A = space.linear_elasticity_matrix(mu, lam)
F = space.source_vector(pde.source, dim=2)
bc = BoundaryCondition(space, dirichlet=pde.dirichlet,  neuman=pde.neuman)
bc.apply_neuman_bc(F, is_neuman_boundary=pde.is_neuman_boundary)
A, F = bc.apply_dirichlet_bc(A, F, uh, is_dirichlet_boundary=pde.is_dirichlet_boundary)
#bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
#A, F = bc.apply_dirichlet_bc(A, F, uh)

uh.T.flat[:] = spsolve(A, F)

error = space.integralalg.L2_error(pde.displacement, uh)
print(error)

node = mesh.entity('node').copy()
uI = space.interpolation(pde.displacement, dim=2)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

fig = plt.figure()
axes = fig.gca()
mesh.node = node + scale*uh
mesh.add_plot(axes)
mesh.node = node + scale*uI
mesh.add_plot(axes, linewidths=0.1, nodecolor='r', edgecolor='r')

plt.show()
