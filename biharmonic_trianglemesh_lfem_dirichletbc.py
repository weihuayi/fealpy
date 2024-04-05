#!/usr/bin/env python3
# 

import argparse
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

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
        val = jnp.column_stack((
            -pi*jnp.sin(pi*x)*jnp.cos(pi*y), 
            -pi*jnp.cos(pi*x)*jnp.sin(pi*y))) #TODO
        return val


    def laplace(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return r

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def laplace_dirichlet(self, p):
        return self.laplace(p);

    def laplace_neuman(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 4*pi**3*(-sin(pi*y)**2 + cos(pi*y)**2)*sin(pi*x)*cos(pi*x) - 8*pi**3*sin(pi*x)*sin(pi*y)**2*cos(pi*x)
        val[..., 1] = 4*pi**3*(-sin(pi*x)**2 + cos(pi*x)**2)*sin(pi*y)*cos(pi*y) - 8*pi**3*sin(pi*x)**2*sin(pi*y)*cos(pi*y)
        return np.sum(val*n, axis=-1) 

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        pi4 = pi**4
        r1 = np.sin(pi*x)**2
        r2 = np.cos(pi*x)**2
        r3 = np.sin(pi*y)**2
        r4 = np.cos(pi*y)**2
        r = 8*pi4*r2*r4 - 16*pi4*r4*r1 - 16*pi4*r2*r3 + 24*pi4*r1*r3
        return r

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

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
A = bform.assembly()

lform = LinearForm(space)
F = ScalarSourceIntegrator(pde.source, q=3)
lform.add_domain_integrator(F)
b = lform.assembly()
