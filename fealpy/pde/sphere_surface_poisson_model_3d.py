#!/usr/bin/python3
'''!    	
	@Author: lq
	@File Name: sphere_surface_poisson_model_3d.py
	@Mail: 2655493031@qq.com.com 
	@Created Time: 2023年09月25日 星期一 16时31分09秒
	@bref 
	@ref 
'''  
import numpy as np
import sympy as sp
from fealpy.decorator import cartesian

class SphereSurfacePDEData:
    def __init__(self, u):

        x, y, z = sp.symbols('x, y, z')

        self.center = [0.0, 0.0, 0.0]
        self.radius = 1.0
        # 定义曲面S的方程 surface_equation
        F = x**2 + y**2 + z**2 - 1

        # 计算真解在曲面S上的梯度
        grad_Fx = sp.diff(F, x)
        grad_Fy = sp.diff(F, y)
        grad_Fz = sp.diff(F, z)
        grad_F = sp.Matrix([grad_Fx, grad_Fy, grad_Fz])
        unit_normal_vector = grad_F/grad_F.norm() # 计算曲面S上的单位法向量

        grad_ux = sp.diff(u, x)
        grad_uy = sp.diff(u, y)
        grad_uz = sp.diff(u, z)
        grad_u = sp.Matrix([grad_ux, grad_uy, grad_uz])

        # 计算梯度在曲面S上的投影，即u在曲面上的梯度
        projection = grad_u - (grad_u.dot(unit_normal_vector)) * unit_normal_vector 

        # 方程右端项 
        f = sp.diff(projection, x) + sp.diff(projection, y) + sp.diff(projection, z) 

        self.u = sp.lambdify((x, y, z), u, "numpy")
        self.f = sp.lambdify((x, y, z), f, "numpy")
        self.gradF = sp.lambdify((x, y, z), grad_F, "numpy")
        self.gradu = sp.lambdify((x, y, z), grad_u, "numpy")
        self.udiff = sp.lambdify((x, y, z), projection, "numpy")


    def surface(self):
        """
        @berif 曲面区域
        """
        return self.center, self.radius

    @cartesian
    def solution(self, p):
        """
        @berif 真解函数
        """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        val = self.u(x, y, z)
        return val

    @cartesian
    def graddient(self, p):
        """
        @berif 真解在曲面S上的梯度 
        """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]

        gradient_su = self.udiff(x, y, z)
        return gradient_su

    @cartesian
    def source(self, p):
        """
        @berif 方程右端项
        """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]

        fval = self.f(x, y, z)
        return fval


# 做一些测试
x, y, z = sp.symbols('x, y, z')
p0 = np.array([1.0, 0.0, 0.0])
p1 = np.array([0.0, 1.0, 0.0])

u = x*y
pde = SphereSurfacePDEData(u)
sdiff0 = pde.graddient(p0)
sdiff1 = pde.graddient(p1)

print('sdiff0:', sdiff0)
print('sdiff1:', sdiff1)
