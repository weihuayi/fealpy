#!/usr/bin/env python3
# 
import argparse
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, spdiags, eye, bmat

import jax 
import jax.numpy as jnp

from fealpy import logger
from fealpy.jax.mesh import TriangleMesh 
from fealpy.jax.functionspace import LagrangeFESpace

from fealpy.jax.fem import ScalarLaplaceIntegrator
from fealpy.jax.fem import BilinearForm

from fealpy.jax.fem import ScalarSourceIntegrator
from fealpy.jax.fem import LinearForm

class CosCosData:
    """
        -\\Delta u = f
        u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self, kappa=1.0):
        self.kappa = kappa # Robin 条件中的系数

    def domain(self):
        """
        @brief 模型定义域
        """
        return np.array([0, 1, 0, 1])

    def solution(self, p):
        """  
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        val = jnp.cos(pi*x)*jnp.cos(pi*y)
        return val # val.shape == x.shape


    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        val = 2*pi*pi*jnp.cos(pi*x)*jnp.cos(pi*y)
        return val#-self.solution(p)

    def gradient(self, p):
        """  
        @brief 真解梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        val = jnp.column_stack((
            -pi*jnp.sin(pi*x)*jnp.cos(pi*y), 
            -pi*jnp.cos(pi*x)*jnp.sin(pi*y)))
        return val # val.shape == p.shape

    def flux(self, p):
        """
        @brief 真解通量
        """
        return -self.gradient(p)

    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)

    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (jnp.abs(y - 1.0) < 1e-12) | (jnp.abs( y -  0.0) < 1e-12)

    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = jnp.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return jnp.abs(x - 1.0) < 1e-12

    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = jnp.sum(grad*n, axis=-1) + self.kappa*self.solution(p) 
        return val

    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return jnp.abs(x - 0.0) < 1e-12

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

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

pde = CosCosData()
domain = pde.domain()

errorType = ['$|| u - u_h||_{\\Omega,0}$', 
        '$||\\nabla u - \\nabla u_h||_{\\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

#ipdb.set_trace()
mesh = TriangleMesh.from_box(box=domain, nx=nx, ny=ny)
space = LagrangeFESpace(mesh, p = p)

bform = BilinearForm(space)
L = ScalarLaplaceIntegrator()
bform.add_domain_integrator(L)
A = bform.assembly()

x = pde.solution(mesh.interpolation_points(p=p))
lform = LinearForm(space)
F = ScalarSourceIntegrator(pde.source, q=p+3)
lform.add_domain_integrator(F)
b = lform.assembly()
print(b)
print(A.toarray())

