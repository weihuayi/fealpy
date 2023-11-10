#!/usr/bin/python3
'''!    	
	@Author: lq
	@File Name: surface_level_set_poisson_model.py
	@Mail: 2655493031@qq.com 
	@Created Time: 2023年11月08日 星期三 16时00分59秒
	@bref 
	@ref 
'''  
import numpy as np
import sympy as sp
from fealpy.decorator import cartesian

class SurfaceLevelSetPDEData:
    def __init__(self, F, u):

        x, y, z = sp.symbols('x, y, z', real=True)

        # 计算曲面上的单位法向量
        grad_Fx = sp.diff(F, x)
        grad_Fy = sp.diff(F, y)
        grad_Fz = sp.diff(F, z)
        grad_F = -sp.Matrix([grad_Fx, grad_Fy, grad_Fz])

        unit_normal_vector = grad_F / grad_F.norm() 

        #计算u在曲面上的梯度
        grad_ux = sp.diff(u, x)
        grad_uy = sp.diff(u, y)
        grad_uz = sp.diff(u, z)
        grad_u = sp.Matrix([grad_ux, grad_uy, grad_uz])
        
        projection = grad_u - (grad_u.dot(unit_normal_vector)) * unit_normal_vector 
        

        # 方程右端项 
        f = sp.diff(projection[0], x) + sp.diff(projection[1], y) + sp.diff(projection[2], z)

        self.F = sp.lambdify((x, y, z), F, "numpy")
        self.u = sp.lambdify((x, y, z), u, "numpy")

        self.f = sp.lambdify((x, y, z), f, "numpy")
        self.gradF = sp.lambdify((x, y, z), grad_F, "numpy")
        self.gradu = sp.lambdify((x, y, z), grad_u, "numpy")
        self.udiff = sp.lambdify((x, y, z), projection, "numpy")

        @cartesian
        def levelset(self, p):
            """
            @berif 曲面的水平集表达式
            """
            x, y, z = p[..., 0], p[..., 1], p[..., 2]

            return self.F(x, y, z)
        @cartesian
        def solution(self, p):
            """
            @berif 真解函数
            """
            x, y, z = p[..., 0], p[..., 1], p[..., 2]

            return self.u(x, y, z)

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

