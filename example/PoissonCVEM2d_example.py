#!/usr/bin/env python3
#

import sys

import numpy as np
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import BoundaryCondition

h = 0.1
maxit = 4
pde = CosCosData()
box = pde.domain()

for i in range(maxit):
    mesh = triangle(box, h/2**(i-1), meshtype='polygon')
    space = ConformingVirtualElementSpace2d(mesh, p=1)
    uh = space.function()
    bc = BoundaryCondition(space, dirichlet=pde.dirichlet)

    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    A, F = bc.apply_dirichlet_bc(A, F, uh)

    uh[:] = spsolve(A, F)
    sh = space.project_to_smspace(uh)

    error = space.integralalg.L2_error(pde.solution, sh)
    print(error)

