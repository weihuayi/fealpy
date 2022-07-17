#!/usr/bin/env python3
# 

"""
    一般椭圆方程的任意次有限元方法。

作者：西安交通大学数学与统计学院 杨迪
说明：FEALPy短课程第三次作业
版本：1.0
日期：31/07/2020
"""

import argparse 
import numpy as np
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import  ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

class PDE:
    """
	Equation:
        -\\nabla\cdot(A(x)\\nabla u + b(x)u) + cu = f in \Omega
	
	B.C.:
	u = g_D on \partial\Omega
	
	Exact Solution:
        u = cos(pi*x)*cos(pi*y)
	
	Coefficients:
	A(x) = [10.0, -1.0; -1.0, 2.0]
	b(x) = [-1; -1]
	c(x) = 1 + x^2 + y^2
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        """ 
		The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
		The right hand side of convection-diffusion-reaction equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y) 
        val += np.cos(pi*x)*np.cos(pi*y)*(x**2 + y**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        return np.array([[10.0, -1.0], [-1.0, 2.0]], dtype=np.float64)

    @cartesian
    def convection_coefficient(self, p):
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x**2 + y**2

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形, 四边形网格上求解一般椭圆问题的任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--mtype',
        default='tri', type=str,
        help='网格类型, 默认为 tri, 即三角形网格, 还可以选择 quad, 即四边形网格.')

parser.add_argument('--ns',
        default=10, type=int,
        help='初始网格 X 与 Y 方向剖分的段数, 默认 10 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
ns = args.ns
maxit = args.maxit
mtype = args.mtype
	
pde = PDE()
domain = pde.domain()
mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype=mtype, p=degree)

errorType = ['$|| u  - u_h ||_0$', '$|| \\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((2, maxit), dtype=mesh.ftype)
NDof = np.zeros(maxit, dtype=mesh.itype)
for i in range(maxit):
    print('Step:', i)
    space = ParametricLagrangeFiniteElementSpace(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()
    uh = space.function() 	# 返回一个有限元函数，初始自由度值全为 0
    A = space.stiff_matrix(c=pde.diffusion_coefficient)
    B = space.convection_matrix(c=pde.convection_coefficient)
    M = space.mass_matrix(c=pde.reaction_coefficient)
    F = space.source_vector(pde.source)
    A += B 
    A += M
    
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value) 
    if i < maxit-1:
        mesh.uniform_refine()
		
# 函数解图像	
uh.add_plot(plt, cmap='rainbow')

# 收敛阶图像
showmultirate(plt, 0, NDof, errorMatrix,  errorType, 
        propsize=40)

# 输出误差的 latex 表格
show_error_table(NDof, errorType, errorMatrix)
plt.show()
