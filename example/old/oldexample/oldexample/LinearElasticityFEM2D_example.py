#!/usr/bin/env python3
# 
import sys

import numpy as np
from scipy.sparse.linalg import spsolve, cg, LinearOperator, spilu
from scipy.sparse import spdiags

import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC, NeumannBC

import pyamg
from timeit import default_timer as timer

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

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i:' % (self.niter))

class LinearElasticityLFEMFastSolver():
    def __init__(self, A, P, isBdDof):
        """
        Notes
        -----

        这里的边界条件处理放到矩阵和向量的乘积运算当中, 所心不需要修改矩阵本身
        """
        self.gdof = P.shape[0]
        self.GD = A.shape[0]//self.gdof

        self.A = A
        self.isBdDof = isBdDof

        # 处理预条件子的边界条件
        bdIdx = np.zeros(P.shape[0], dtype=np.int_)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, P.shape[0], P.shape[0])
        T = spdiags(1-bdIdx, 0, P.shape[0], P.shape[0])
        P = T@P@T + Tbd
        self.ml = pyamg.ruge_stuben_solver(P) 

    def preconditioner(self, b):
        GD = self.GD
        b = b.reshape(GD, -1)
        r = np.zeros_like(b)
        for i in range(GD):
            r[i] = self.ml.solve(b[i], tol=1e-8, accel='cg')       
        return r.reshape(-1)

    def solve(self, uh, F, tol=1e-8):
        """

        Notes
        -----
        uh 是初值, uh[isBdDof] 中的值已经设为 D 氏边界条件的值, uh[~isBdDof]==0.0
        """

        GD = self.GD
        gdof = self.gdof

        counter = IterationCounter()
        P = LinearOperator((GD*gdof, GD*gdof), matvec=self.preconditioner)
        uh.T.flat, info = cg(self.A, F.T.flat, x0= uh.T.flat, M=P, tol=1e-8,
                callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of pcg:", counter.niter)

        return uh 

n = int(sys.argv[1])
p = int(sys.argv[2])
scale = float(sys.argv[3])

pde = BoxDomain2DData()

mesh = pde.init_mesh(n=n)

area = mesh.entity_measure('cell')

space = LagrangeFiniteElementSpace(mesh, p=p)

bc0 = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
bc1 = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)

uh = space.function(dim=2) # (gdof, 2) and vector fem function uh[i, j] 
P = space.stiff_matrix(c=2*pde.mu)
A = space.linear_elasticity_matrix(pde.lam, pde.mu) # (2*gdof, 2*gdof)
F = space.source_vector(pde.source, dim=2) 
F = bc1.apply(F)
A, F = bc0.apply(A, F, uh)

print(A.shape)


if False:
    uh.T.flat[:] = spsolve(A, F) # (2, gdof ).flat
elif False:
    N = len(F)
    print(N)
    ilu = spilu(A.tocsc(), drop_tol=1e-6, fill_factor=40)
    M = LinearOperator((N, N), lambda x: ilu.solve(x))
    start = timer()
    uh.T.flat[:], info = cg(A, F, tol=1e-8, M=M)   # solve with CG
    print(info)
    end = timer()
    print('time:', end - start)
else:
    isBdDof = space.set_dirichlet_bc(pde.dirichlet, uh,
            threshold=pde.is_dirichlet_boundary)
    solver = LinearElasticityLFEMFastSolver(A, P, isBdDof) 
    start = timer()
    uh[:] = solver.solve(uh, F) 
    end = timer()
    print('time:', end - start)


# 原始的网格
mesh.add_plot(plt)

# 变形的网格
#mesh.node += scale*uh
#mesh.add_plot(plt)
#plt.show()
