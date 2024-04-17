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
from fealpy.jax.functionspace import InteriorPenaltyLagrangeFESpace2d

from fealpy.jax.fem import ScalarBiharmonicIntegrator
from fealpy.jax.fem import ScalarInteriorPenaltyIntegrator
from fealpy.jax.fem import BilinearForm

from fealpy.jax.fem import ScalarSourceIntegrator
from fealpy.jax.fem import LinearForm
jax.config.update("jax_enable_x64", True)

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
    f = f - A@uh.reshape(-1)
    bdIdx = jnp.zeros(A.shape[0], dtype=np.int_)
    bdIdx = bdIdx.at[isDDof.reshape(-1)].set(1)
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

mesh  = TriangleMesh.from_box(box=domain, nx=nx, ny=ny)
space = InteriorPenaltyLagrangeFESpace2d(mesh, p = p)

errorType = ['$|| Ax-b ||_{\\Omega,0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):

    bform = BilinearForm(space)
    L = ScalarBiharmonicIntegrator()

    bform.add_domain_integrator(L)
    A0 = bform.assembly()
    
    P0 = ScalarInteriorPenaltyIntegrator(gamma=0.01)
    P  = P0.assembly_face_matrix(space)  
    A  = A0 + P

    lform = LinearForm(space)
    F = ScalarSourceIntegrator(pde.source, q=p+3)
    lform.add_domain_integrator(F)
    b = lform.assembly()
    
    x = pde.solution(mesh.interpolation_points(p=p))
    
    Bd = pde.is_boundary_dof(mesh.interpolation_points(p=p))
    gd = jnp.zeros_like(x)
    gd = gd.at[Bd].set(x[Bd])
    
    A, f = apply_dbc(A, b, gd, pde.is_boundary_dof(mesh.interpolation_points(p=p)))

    uh, tol = cg(A, f, atol=1e-15)

    errorMatrix[0, i] = jnp.linalg.norm(uh-x)
    errorMatrix[1, i] = jnp.linalg.norm(A@x-f)
    print(errorMatrix)

    if i < maxit-1:
        nx = nx*2
        ny = ny*2
        mesh = TriangleMesh.from_box(box=domain, nx=nx, ny=ny)
        space = InteriorPenaltyLagrangeFESpace2d(mesh, p = p)


print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

