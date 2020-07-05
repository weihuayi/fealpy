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
        val = np.array([500, 0.0], dtype=np.float64)
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

class CrackBoxDomain2DData():
    def __init__(self, E=1e+5, nu=0.2):
        self.E = E 
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

    def domain(self):
        return [0, 1, 0, 1]

    def init_mesh(self, n=3, meshtype='tri'):
        node = np.array([
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0, 0.5),
            (0.0, 1.0),
            (0.5, 1.0),
            (0.5, 1.0),
            (1.0, 1.0)], dtype=np.float)
        cell = np.array([
            (3, 0, 4), (1, 4, 0),
            (4, 1, 5), (2, 5, 1),
            (6, 3, 7), (4, 7, 3),
            (8, 4, 9), (5, 9, 4)], dtype=np.int)
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
        val0 = np.array([500, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        #val1 = np.array([-500, 0.0], dtype=np.float64)
        return val0.reshape(shape)

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

class CrossCrackBoxDomain2DData():
    def __init__(self, E=1e+5, nu=0.2):
        self.E = E 
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

    def domain(self):
        return [0, 1, 0, 1]

    def init_mesh(self, n=0, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(2)

        NN = mesh.number_of_nodes()
        node = np.zeros((NN+3, 2), dtype=np.float64)
        node[:NN] = mesh.entity('node')
        node[NN:] = node[[5], :]
        cell = mesh.entity('cell')

        cell[13][cell[13] == 5] = NN
        cell[18][cell[18] == 5] = NN

        cell[19][cell[19] == 5] = NN+1
        cell[12][cell[12] == 5] = NN+1

        cell[6][cell[6] == 5] = NN+2
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


        if False:
            x = p[..., 0]
            y = p[..., 1]
            val = np.zeros_like(p)
            flag0 = np.abs(x-1) < 1e-13 
            val[:, 0][flag0] = 500

            flag1 = (np.abs(x - 0.5) < 1e-13) | (np.abs(y - 0.5) < 1e-13)

            val[flag1, 0] = -500*n[None, :, 0]
            val[flag1, 1] = -500*n[None, :, 1]
        if True:
            x = p[..., 0]
            y = p[..., 1]
            val = np.zeros_like(p)

            flag0 = np.abs(x-1) < 1e-13 
            val[:, 0][flag0] = 500

            flag1 = (x > 0.25) & (x < 0.75) & (np.abs(y-0.5) < 1e-13)
            val[:, 1][flag1] = -np.arra*n[None, ...]


            return val.reshape(shape)

        if False:
            val0 = np.array([500, 0.0], dtype=np.float64)
            shape = len(p.shape[:-1])*(1, ) + (2, )
            return val0.reshape(shape)

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



# pde = BoxDomain2DData()
#pde = CrackBoxDomain2DData()
pde = CrossCrackBoxDomain2DData()

mu = pde.mu
lam = pde.lam
mesh = pde.init_mesh(n=n)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

space = LagrangeFiniteElementSpace(mesh, p=p)
bc0 = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
bc1 = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)

uh = space.function(dim=2)
A = space.linear_elasticity_matrix(mu, lam)
F = space.source_vector(pde.source, dim=2)
bc1.apply(F)
A, F = bc0.apply(A, F, uh)
uh.T.flat[:] = spsolve(A, F)

scale = np.arange(1.0, scale, 0.1)
node = mesh.entity('node')
fname = 'test'
for val in scale:
    mesh.node = node + val*uh
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.savefig(fname + str(val) + '.png')

plt.close()
