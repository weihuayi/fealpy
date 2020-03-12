#!/usr/bin/env python3
#

import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import BoundaryCondition

node = 0.5*np.array([
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
    (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
    (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float)
cell = np.array([0, 3, 4, 4, 1, 0,
    1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int)
cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int)
mesh = PolygonMesh(node, cell, cellLocation)

if False:
    node = np.array([
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
    cell = np.array([0, 1, 2, 3], dtype=np.int)
    cellLocation = np.array([0, 4], dtype=np.int)
    mesh = PolygonMesh(node, cell, cellLocation)

space = ConformingVirtualElementSpace2d(mesh, p=1)
A0, S0 = space.chen_stability_term()
A = space.stiff_matrix()

print("A:", A.toarray())
print("A0:", A0.toarray())
print("S0:", S0.toarray())

np.savetxt('A.txt', A.toarray(), fmt='%.2e')
np.savetxt('A0.txt', A0.toarray(), fmt='%.2e')
np.savetxt('S0.txt', S0.toarray(), fmt='%.2e')

if False:
    h = 0.1
    maxit = 1
    pde = CosCosData()
    box = pde.domain()

    for i in range(maxit):
        mesh = triangle(box, h/2**(i-1), meshtype='polygon')
        space = ConformingVirtualElementSpace2d(mesh, p=1)
        uh = space.function()
        bc = BoundaryCondition(space, dirichlet=pde.dirichlet)

        A0, S0 = space.chen_stability_term()
        A0 = A0.toarray()
        S0 = S0.toarray()

        A = space.stiff_matrix()
        F = space.source_vector(pde.source)

        A, F = bc.apply_dirichlet_bc(A, F, uh)

        uh[:] = spsolve(A, F)
        sh = space.project_to_smspace(uh)

        error = space.integralalg.L2_error(pde.solution, sh)
        print(error)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
