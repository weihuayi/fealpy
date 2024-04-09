#!/usr/bin/env python3
# 

import argparse
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, cg, lgmres
import scipy.sparse as sp
from jax.scipy.linalg import solve
from scipy.sparse import csr_matrix, spdiags, eye, bmat

import jax 
import jax.numpy as jnp

from fealpy import logger
from fealpy.jax.mesh import TriangleMesh 
from fealpy.jax.functionspace import LagrangeFESpace

from fealpy.jax.fem import ScalarBiharmonicIntegrator
from fealpy.jax.fem import BilinearForm

from fealpy.jax.fem import ScalarSourceIntegrator
from fealpy.jax.fem import LinearForm

class SinSinData:

    def domain(self):
        return [0, 1, 0, 1]

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        r = jnp.sin(pi*x)*jnp.sin(pi*y)/pi**4/4.0
        return r

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val = jnp.column_stack((
            jnp.cos(pi*x)*jnp.sin(pi*y)/pi**3/4.0, 
            jnp.sin(pi*x)*jnp.cos(pi*y)/pi**3/4.0)) 
        return val

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)
    
    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return jnp.sum(val*n, axis=-1)

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        return jnp.sin(pi*x)*jnp.sin(pi*y)

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)


def apply_dbc(A, f, uh, isDDof):
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isDDof.reshape(-1)] = 1
    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0@A@D0 + D1
    
    f = f.at[isDDof.reshape(-1)].set(uh[isDDof].reshape(-1))
    return A, f

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次内罚有限元方法求解双调和方程
        """)

parser.add_argument('--degree',
        default=2, type=int,
        help='Lagrange 有限元空间的次数, 默认为 2 次.')

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

pde = SinSinData()
domain = pde.domain()

mesh = TriangleMesh.from_box(box=domain, nx=nx, ny=ny)
space = LagrangeFESpace(mesh, p = p)

errorType = ['$|| Ax-b ||_{\\Omega,0}$']
errorMatrix = np.zeros((4, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):

    bform = BilinearForm(space)
    L = ScalarBiharmonicIntegrator()


    bform.add_domain_integrator(L)
    A0 = bform.assembly()
    
    P = L.penalty_matrix(space, gamma=100)
    A = A0 + P

    lform = LinearForm(space)
    F = ScalarSourceIntegrator(pde.source, q=p+3)
    lform.add_domain_integrator(F)
    b = lform.assembly()
    
    x = pde.solution(interpolation_points(p=p))
    
    A, f = apply_dbc(A, b, x, pde.is_boundary_dof(interpolation_points(p=p)))

    node = mesh.entity('node')
    print('自由度个数:', f.shape)
    print('A的秩：', jnp.linalg.matrix_rank(A.toarray()))
#    uh = spsolve(A, b)
    
    uh, tol = cg(A, b, atol=1e-15)
#    print('真解：', uh, tol)
#    print(b)
#    print(A.toarray())
    errorMatrix[0, i] = jnp.linalg.norm(uh-x)
    errorMatrix[1, i] = jnp.linalg.norm(A@x-f)
    errorMatrix[2, i] = jnp.linalg.norm(A0@x-f)
    errorMatrix[3, i] = jnp.linalg.norm(P@x-f)
    

    if i < maxit-1:
        nx = nx*2
        ny = ny*2
        mesh = TriangleMesh.from_box(box=domain, nx=nx, ny=ny)
        space = LagrangeFESpace(mesh, p = p)


    print(errorMatrix[0, i])
print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

