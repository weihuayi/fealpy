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

print('test')
mesh = pde.init_mesh(n=7, meshtype='tri')
print('test')
space = LagrangeFiniteElementSpace(mesh, p=1)
print('test')

A = space.stiff_matrix()
print('test')
b = space.source_vector(pde.source)
print('test')
uh = space.function()
bc = BoundaryCondition(space, robin=pde.robin)
bc.apply_robin_bc(A, b)
uh[:] = spsolve(A, b).reshape(-1)
error = space.integralalg.L2_error(pde.solution, uh)
print(error)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

