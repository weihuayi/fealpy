#!/usr/bin/env python3
# 

__doc__= """
作者：西安交通大学数学与统计学院 杨迪
说明：FEALPy短课程第三次作业
版本：1.0
日期：31/07/2020
"""

import sys
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

class CDRMODEL:
    """
	Equation:
    -\\nabla\cdot(A(x)\\nabla u + b(x)u) + cu = f in \Omega
	
	B.C.:
	u = g_D on \partial\Omega
	
	Exact Solution:
        u = cos(pi*x)*cos(pi*y)
	
	Coefficients:
	A(x) = [10.0, -1.0;
			-1.0, 2.0]
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

	
p = int(sys.argv[1])		# Lagrange 有限元多项式次数
n = int(sys.argv[2])		# 初始网格剖分段数
maxit = int(sys.argv[3])	# 网格加密最大次数

pde = CDRMODEL()
domain = pde.domain()


mesh = MF.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')

NDof = np.zeros(maxit, dtype=mesh.itype)
errorMatrix = np.zeros((2, maxit), dtype=mesh.ftype)
errorType = ['$|| u  - u_h ||_0$', '$|| \\nabla u - \\nabla u_h||_0$']
for i in range(maxit):
    print('Step:', i)
    space = LagrangeFiniteElementSpace(mesh, p=p)
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

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value, power=2)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value,
            power=2)

    if i < maxit-1:
        mesh.uniform_refine()
		
# 函数解图像	
uh.add_plot(plt, cmap='rainbow')

# 收敛阶图像
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=10)

# 输出误差的 latex 表格
show_error_table(NDof, errorType, errorMatrix)
plt.show()
