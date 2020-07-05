#!/usr/bin/env python3
# 

import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition


pde = CosCosData()

n = int(sys.argv[1])
mesh = pde.init_mesh(n=n, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh, p=1)


A = space.stiff_matrix()
b = space.source_vector(pde.source)
uh = space.function()
#bc = BoundaryCondition(space, robin=pde.robin)
A, b = space.set_robin_bc(A, b, pde.robin)
#A, b = bc.apply_robin_bc(A, b)
uh[:] = spsolve(A, b).reshape(-1)
error = space.integralalg.L2_error(pde.solution, uh)
print(error)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes,showindex=True)

plt.show()

