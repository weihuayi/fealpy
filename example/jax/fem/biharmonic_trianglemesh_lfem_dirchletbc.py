#!/usr/bin/env python3
# 

import argparse
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from jax.scipy.linalg import solve

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

def interpolation_points(p: int, index=np.s_[:]):
    """
    @brief 获取三角形网格上所有 p 次插值点
    """
    cell = mesh.entity('cell')
    node = mesh.entity('node')

    NN = mesh.number_of_nodes()
    GD = mesh.geo_dimension()

    gdof = mesh.number_of_global_ipoints(p)
    ipoints = jnp.zeros((gdof, GD), dtype=jnp.float_)
    ipoints = ipoints.at[:NN, :].set(node)


    NE = mesh.number_of_edges()

    edge = mesh.entity('edge')

    w = jnp.zeros((p-1, 2), dtype=jnp.float_)

    w = w.at[:, 0].set(jnp.arange(p-1, 0, -1) / p)

    w = w.at[:, 1].set(w[-1::-1, 0])

    ipoints = ipoints.at[NN:NN+(p-1)*NE, :].set(
        jnp.einsum('ij, ...jm->...im', w, node[edge, :]).reshape(-1, GD)
    )

    return ipoints # (gdof, GD)

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

bform = BilinearForm(space)
L = ScalarBiharmonicIntegrator()


bform.add_domain_integrator(L)
A0 = bform.assembly()
P = L.penalty_matrix(space, gamma=10)
A = A0 + P

lform = LinearForm(space)
F = ScalarSourceIntegrator(pde.source, q=3)
lform.add_domain_integrator(F)
b = lform.assembly()

node = mesh.entity('node')

print(b)
print(A.toarray())
gdof = space.dof.number_of_global_dofs()
u = jnp.zeros(gdof)
x = pde.solution(interpolation_points(p=2))

print(np.max(np.abs(A@x-b)))


