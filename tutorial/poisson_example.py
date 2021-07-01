#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.boundarycondition import NeumannBC 
from fealpy.boundarycondition import RobinBC

from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve

class CosCosData:
    """
    -\Delta u + 3u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)
        val *= np.cos(pi*y)
        val *= 2*pi*pi + 3
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        eps = 1e-12
        y = p[..., 1]
        return (np.abs(y - 1.0) < eps) | ( np.abs(y - 0.0) < eps)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        eps = 1e-12
        x = p[..., 0]
        return np.abs(x - 1.0) < eps

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa

    @cartesian
    def is_robin_boundary(self, p):
        eps = 1e-12
        x = p[..., 0]
        return np.abs( x - 0.0) < eps 


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上任意次有限元方法        
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--ns',
        default=10, type=int,
        help='初始网格 x 和 y 方向的剖分段数, 默认 10 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
ns = args.ns
maxit = args.maxit

pde = CosCosData()
box = pde.domain()
mesh = MF.boxmesh2d(box, nx=ns, ny=ns, meshtype='tri')

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = LagrangeFiniteElementSpace(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()
    bc0 = DirichletBC(space, pde.dirichlet, pde.is_dirichlet_boundary) 
    bc1 = NeumannBC(space, pde.neumann, pde.is_neumann_boundary)
    bc2 = RobinBC(space, pde.robin, pde.is_robin_boundary)


    A = space.stiff_matrix() + 3*space.mass_matrix()
    F = space.source_vector(pde.source)

    F = bc1.apply(F)
    A, F = bc2.apply(A, F)

    uh = space.function()
    A, F = bc0.apply(A, F, uh)
    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()



fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
uh.add_plot(axes, cmap='rainbow')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

plt.show()
