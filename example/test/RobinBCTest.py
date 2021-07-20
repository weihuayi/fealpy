#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat


import scipy.io as sio
import matplotlib.pyplot as plt


from fealpy.decorator import cartesian
from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace

class XYData:
    """
    -\Delta u = f
    u = x*y 
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype == 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'squad':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float)
        """
        x = p[..., 0]
        y = p[..., 1]
        val = x*y 
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        f = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return f.reshape(shape) 

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = y 
        val[..., 1] = x 
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

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
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float).reshape(shape)
        val += self.solution(p) 
        return val, kappa


class RobinBCTest():
    def __init__(self):
        pass

    def solve_poisson_robin(self, p=1, n=1, plot=True):

        pde = XYData()
        mesh = pde.init_mesh(n=n)

        node = mesh.node
        cell = mesh.entity("cell")
        name = 'RobinBCTest.mat'
        space = LagrangeFiniteElementSpace(mesh, p=p)
        A = space.stiff_matrix()
        F = space.source_vector(pde.source)
#        print(A.toarray())

#        A, F = space.set_robin_bc(A, F, pde.robin)
 
        uh = space.function()
        #bc = BoundaryCondition(space, robin=pde.robin)
        A, b = space.set_robin_bc(A, F, pde.robin)
        uh[:] = spsolve(A, b).reshape(-1)
        error = space.integralalg.L2_error(pde.solution, uh)
        print(error)

#        print('A:', A.toarray())
#        print('F:', F)


        if plot:
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()


test = RobinBCTest()

if sys.argv[1] == 'solve_poisson_robin':
    p = int(sys.argv[2])
    n = int(sys.argv[3])
    test.solve_poisson_robin(p=p, n=n)

