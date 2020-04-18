#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from fealpy.pde.poisson_2d import  CosCosData 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition

from fealpy.graph import metis


n = int(sys.argv[1])
m = int(sys.argv[2])

p=1

pde = CosCosData() 

mesh = pde.init_mesh(n=n)
space = LagrangeFiniteElementSpace(mesh, p=p)

bc = BoundaryCondition(space, dirichlet=pde.dirichlet)
uh = space.function()
A = space.stiff_matrix()
F = space.source_vector(pde.source)
A, F = bc.apply_dirichlet_bc(A, F, uh)
uh[:] = spsolve(A, F)

error = space.integralalg.L2_error(pde.solution, uh)

edgecuts, parts = metis.part_mesh(mesh, nparts=m, entity='cell', contig=True)
print(parts)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#mesh.find_node(axes, color=parts)
mesh.find_cell(axes, color=parts)
plt.show()
