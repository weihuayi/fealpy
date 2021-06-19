#!/usr/bin/env python3
# 

import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import BoundaryCondition
from fealpy.decorator  import cartesian 
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import LagrangeFiniteElementSpace


class CosCosCosData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='tet'):
        node = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        mesh = TetrahedronMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    @cartesian
    def solution(self, p):
        """ the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -pi*sin(pi*x)*cos(pi*y)*cos(pi*z)
        val[..., 1] = -pi*cos(pi*x)*sin(pi*y)*cos(pi*z)
        val[..., 2] = -pi*cos(pi*x)*cos(pi*y)*sin(pi*z)
        return val

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        """
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float).reshape(shape)
        val += self.solution(p) 
        return val, kappa

class RobinBCTest():
    def __init__(self):
        pass

    def solve_poisson_robin(self, p=1, n=1, plot=True):

        pde = CosCosCosData()
        mesh = pde.init_mesh(n=n)

        space = LagrangeFiniteElementSpace(mesh, p=p)

        A = space.stiff_matrix()
        F = space.source_vector(pde.source) 
        uh = space.function()
        bc = BoundaryCondition(space, robin=pde.robin)
        A, b = space.set_robin_bc(A, F, pde.robin)
        uh[:] = spsolve(A, b).reshape(-1)
        error = space.integralalg.L2_error(pde.solution, uh)
        print(error)


test = RobinBCTest()

if sys.argv[1] == 'solve_poisson_robin':
    p = int(sys.argv[2])
    n = int(sys.argv[3])
    test.solve_poisson_robin(p=p, n=n)
