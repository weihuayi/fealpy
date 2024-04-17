#!/usr/bin/env python3
# 

import argparse
import ipdb
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve, cg, lgmres
from scipy.sparse import csr_matrix, spdiags, eye, bmat

from fealpy import logger
from fealpy.mesh import TriangleMesh 
from fealpy.functionspace import InteriorPenaltyBernsteinFESpace2d

from fealpy.fem import ScalarBiharmonicIntegrator
from fealpy.fem import ScalarInteriorPenaltyIntegrator
from fealpy.fem import BilinearForm

from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm

from pde import DoubleLaplacePDE 

class SinSinData:

    def domain(self):
        return [0, 1, 0, 1]

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        val = sin(2*pi*x)**2*sin(2*pi*y)**2
        return val 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        cos = np.cos
        val0 = 4*pi*sin(2*pi*x)*sin(2*pi*y)**2*cos(2*pi*x)
        val1 = 4*pi*sin(2*pi*x)**2*sin(2*pi*y)*cos(2*pi*y)
        return np.concatenate([val0[..., None], val1[..., None]], axis=-1) 

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        cos = np.cos
        val0 = -8*pi**2*sin(2*pi*x)**2*sin(2*pi*y)**2 + 8*pi**2*sin(2*pi*y)**2*cos(2*pi*x)**2
        val1 = 16*pi**2*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*x)*cos(2*pi*y)
        val2 = -8*pi**2*sin(2*pi*x)**2*sin(2*pi*y)**2 + 8*pi**2*sin(2*pi*x)**2*cos(2*pi*y)**2
        return np.concatenate([val0[..., None], val1[..., None], val1[...,
            None], val2[..., None]], axis=-1).reshape(p.shape+(2, 2))

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)
    
    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        cos = np.cos
        val = 64*pi**4*(3*sin(2*pi*x)**2*sin(2*pi*y)**2 - 3*sin(2*pi*x)**2*cos(2*pi*y)**2 - sin(2*pi*y)**2*cos(2*pi*x)**2 + cos(2*pi*x)**2*cos(2*pi*y)**2) + 64*pi**4*(3*sin(2*pi*x)**2*sin(2*pi*y)**2 - sin(2*pi*x)**2*cos(2*pi*y)**2 - 3*sin(2*pi*y)**2*cos(2*pi*x)**2 + cos(2*pi*x)**2*cos(2*pi*y)**2)
        return val

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

def is_boundary_dof(p):
    eps = 1e-14 
    return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)


def apply_dbc(A, f, uh, isDDof):
    f = f - A@uh.reshape(-1)
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isDDof.reshape(-1)] = 1
    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0@A@D0 + D1
    
    f[isDDof.reshape(-1)] = uh[isDDof].reshape(-1)
    return A, f

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次内罚有限元方法求解双调和方程
        """)

parser.add_argument('--degree',
        default=2, type=int,
        help='Bernstein 有限元空间的次数, 默认为 2 次.')

parser.add_argument('--nx',
        default=1, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=1, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

x = sp.symbols("x")
y = sp.symbols("y")
u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x))**2
pde = DoubleLaplacePDE(u)

mesh  = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
space = InteriorPenaltyBernsteinFESpace2d(mesh, p = p)

errorType = ['$|| Ax-b ||_{\\Omega,0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)
print(mesh.entity('cell'))
print(mesh.entity('edge'))

for i in range(maxit):

    bform = BilinearForm(space)
    L = ScalarBiharmonicIntegrator()

    bform.add_domain_integrator(L)
    A0 = bform.assembly()
    
    P0 = ScalarInteriorPenaltyIntegrator(gamma=0.01)
    P  = P0.assembly_face_matrix(space)  
    A  = A0 + P

    lform = LinearForm(space)
    F = ScalarSourceIntegrator(pde.source, q=p+2)
    lform.add_domain_integrator(F)
    b = lform.assembly()
    
    x = pde.solution(mesh.interpolation_points(p=p))
    
    Bd = is_boundary_dof(mesh.interpolation_points(p=p))
    gd = np.zeros_like(x)
    gd[Bd] = x[Bd]
    
    A, f = apply_dbc(A, b, gd, is_boundary_dof(mesh.interpolation_points(p=p)))

    #uh, tol = cg(A, f, atol=1e-10)
    uh = space.function()
    uh[:] = spsolve(A, f)
    print("AAA : ", np.max(A0.data))
    print("AAA : ", np.max(P.data))

    errorMatrix[0, i] = mesh.error(uh, pde.solution)
    errorMatrix[1, i] = np.max(A@x-f)
    print(errorMatrix)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    cell = np.array(cell)

    NN = len(node)
    node0 = np.array(node)
    node0 = np.c_[node0, x[:NN, None]]
    node1 = np.array(node)
    node1 = np.c_[node1, uh[:NN, None]]
    meshv0 = TriangleMesh(node0, cell)
    meshv1 = TriangleMesh(node1, cell)
    meshv0.to_vtk(fname='aaa.vtu')
    meshv1.to_vtk(fname='bbb.vtu')

    if i < maxit-1:
        nx = nx*2
        ny = ny*2
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
        space = InteriorPenaltyBernsteinFESpace2d(mesh, p = p)


print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

