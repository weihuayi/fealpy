#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh 
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DiffusionIntegrator 
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC

import ipdb

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        QuadrangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=8, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=8, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

#ipdb.set_trace()
pde = CosCosData()
domain = pde.domain()

mesh = QuadrangleMesh.from_box(domain, nx=nx, ny=ny)

errorType = ['$|| u - u_h||_{\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator(q=p+2))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=p+2))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = mesh.error(pde.solution, uh, q=p+2)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+2)

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
