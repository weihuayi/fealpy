import argparse

import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData
from pde import LaplacePDE

from fealpy.mesh.triangle_mesh import TriangleMesh 

from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace

from fealpy.fem.diffusion_integrator import DiffusionIntegrator 
from fealpy.fem.scalar_source_integrator import ScalarSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC

from fealpy.solver import GAMGSolver


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

x = sp.symbols('x')
y = sp.symbols('y')
u = x**4*y**5#(sp.sin(4*sp.pi*x)*sp.sin(4*sp.pi*y))**6
pde = LaplacePDE(u)
domain =[0,1,0,1] #pde.domain()

mesh = TriangleMesh.from_box(box = domain, nx = nx, ny = ny)

errorType = ['$|| u - u_h||_{\\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    space = LagrangeFESpace(mesh, p = p, spacetype = 'C', doforder = 'vdims')
    NDof[i] = space.number_of_global_dofs()

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator(q = p+2))
    A = bform.assembly()


    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(f = pde.source, q = p+2))
    F = lform.assembly()

    bc = DirichletBC(space = space, gD = pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)

    solver = GAMGSolver(ptype='W', sstep=2)
    solver.setup(A)
    uh[:] = solver.solve(F)
    #uh[:] = spsolve(A, F)

    errorMatrix[0, i] = mesh.error(pde.solution, uh)
    errorMatrix[1, i] = mesh.error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
