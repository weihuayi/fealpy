"""
@brief 二维的扩散+对流+反应项的模型数据
@author Huayi Wei <weihuayi@xtu.edu.cn>

TODO: 增加 sympy 自动生成测试 PDE 模型
"""
import numpy as np
from fealpy.decorator import cartesian, barycentric

class PDEData_0:
    """
	Equation:
        -\Delta u + b\cdot\nabla u + u = f in \Omega
	
	B.C.:
	u = g_D on \partial\Omega
	
	Exact Solution:
        u = cos(pi*x)*cos(pi*y)
	
	Coefficients:
	b(x) = [-1.0; -1.0]
    """
    def __init__(self, kappa=1.0):
        self.kappa = kappa 

    def domain(self):
        """
        @brief 模型定义域
        """
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        """ 
        @brief 真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (2*pi**2+1)*self.solution(p)
        val+= pi*np.sin(pi*x)*np.cos(pi*y)
        val+= pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ 
        @brief 真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (np.abs(y - 1.0) < 1e-12) | (np.abs( y -  0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return np.abs(x - 1.0) < 1e-12

    @cartesian
    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return np.abs(x - 0.0) < 1e-12

class PDEData_1:
    """
	Equation:
        -\Delta u + b\cdot\nabla u + u = f in \Omega
	
	B.C.:
	u = g_D on \partial\Omega
	
	Exact Solution:
        u = sin(pi*x)*sin(pi*y)
	
	Coefficients:
	b(x) = [-1.0; -1.0]
    """
    def __init__(self, kappa=1.0):
        self.kappa = kappa 

    def domain(self):
        """
        @brief 模型定义域
        """
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        """ 
        @brief 真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (2*pi**2+1)*self.solution(p)
        val-= pi*np.cos(pi*x)*np.sin(pi*y)
        val-= pi*np.sin(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ 
        @brief 真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (np.abs(y - 1.0) < 1e-12) | (np.abs( y -  0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return np.abs(x - 1.0) < 1e-12

    @cartesian
    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return np.abs(x - 0.0) < 1e-12

class PDEData_2:
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
    def __init__(self, kappa=1.0):
        self.kappa = kappa 

    def domain(self):
        """
        @brief 模型定义域
        """
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        """ 
        @brief 真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
        @brief 源项
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
        @brief 真解的梯度
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
        """
        @brief 扩散系数
        """
        return np.array([[10.0, -1.0], [-1.0, 2.0]], dtype=np.float64)

    @cartesian
    def convection_coefficient(self, p):
        """
        @brief 对流系数
        """
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        """
        @brief 反应系数
        """
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x**2 + y**2

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (np.abs(y - 1.0) < 1e-12) | (np.abs( y -  0.0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        """ 
        @brief Neumann 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        """
        @brief Neumann 边界的判断函数
        """
        x = p[..., 0]
        return np.abs(x - 1.0) < 1e-12

    @cartesian
    def robin(self, p, n):
        """
        @brief Robin 边界条件
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        """
        @brief Robin 边界条件判断函数
        """
        x = p[..., 0]
        return np.abs(x - 0.0) < 1e-12
