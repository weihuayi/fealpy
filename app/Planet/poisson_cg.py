#!/usr/bin/env python3
# 

"""
Lagrange 元求解 Poisson 方程, 

.. math::
    -\Delta u = f

转化为

.. math::
    (\\nabla u, \\nabla v) = (f, v)

Notes
-----

"""

import numpy as np

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.pde.poisson_3d import CosCosCosData as PDE
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 

from fealpy.decorator import timer

# solver
from scipy.sparse.linalg import cg, LinearOperator

pde = PDE()
mesh = pde.init_mesh(n=1)

space = LagrangeFiniteElementSpace(mesh, p=1)
bc = DirichletBC(space, pde.dirichlet) 

uh = space.function()
A = space.stiff_matrix()
F = space.source_vector(pde.source)
A, F = bc.apply(A, F, uh)

def linear_operator(b):
    b = A@b
    return b

@timer
def solve(u):
    dof = F.shape[0]

    P = LinearOperator((dof, dof), matvec=linear_operator)

    u.T.flat, info = cg(P, F, tol=1e-8)
    return u

uh[:] = solve(uh)

