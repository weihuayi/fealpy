#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC, NeumannBC

class BoxDomain2DData():
    def __init__(self, E=1e+5, nu=0.2):
        self.E = E 
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

    def domain(self):
        return [0, 1, 0, 1]

    def init_mesh(self, n=3, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh 

    @cartesian
    def displacement(self, p):
        return 0.0

    @cartesian
    def jacobian(self, p):
        return 0.0

    @cartesian
    def strain(self, p):
        return 0.0

    @cartesian
    def stress(self, p):
        return 0.0

    @cartesian
    def source(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def neumann(self, p, n):
        val = np.array([-500, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x) < 1e-13
        return flag

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x - 1) < 1e-13
        return flag

    @cartesian
    def is_fracture_boundary(self, p):
        pass


n = int(sys.argv[1])
p = int(sys.argv[2])
scale = float(sys.argv[3])

pde = BoxDomain2DData()

mesh = pde.init_mesh(n=n)


space = LagrangeFiniteElementSpace(mesh, p=p)

bc0 = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
bc1 = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)

uh = space.function(dim=2) # (gdof, 2) and vector fem function uh[i, j] 
A = space.linear_elasticity_matrix(pde.mu, pde.lam) # (2*gdof, 2*gdof)
F = space.source_vector(pde.source, dim=2) 
F = bc1.apply(F)
A, F = bc0.apply(A, F, uh)
uh.T.flat[:] = spsolve(A, F) # (2, gdof ).flat

# 原始的网格
mesh.add_plot(plt)

# 变形的网格
mesh.node += scale*uh
mesh.add_plot(plt)

plt.show()
