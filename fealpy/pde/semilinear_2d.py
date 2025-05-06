
from ..backend import backend_manager as bm
from ..decorator import cartesian
from typing import Sequence

PI = bm.pi
class SinSinData():
    def __init__(self):
        pass
    def domain(self):
        return [0, 1, 0, 2]
    #扩散项系数
    @cartesian
    def diffusion_coefficient(self, p):

        return 10

    @cartesian
    def diffusion_coefficient_right(self, p):

        return -self.diffusion_coefficient(p)

    #反应项系数
    @cartesian
    def reaction_coefficient(self, p):

        return 1

    @cartesian
    def reaction_coefficient_right(self, p):

        return -self.reaction_coefficient(p)

    #真解
    @cartesian
    def solution(self, p):

        x = p[..., 0]
        y = p[..., 1]
        sol = bm.zeros(p.shape[:-1], dtype=bm.float64)
        sol[:] = bm.sin(PI * x) * bm.sin(PI * y)
        return sol

    #真解的梯度
    @cartesian
    def gradient(self, p):

        x = p[..., 0]
        y = p[..., 1]
        grad = bm.zeros(p.shape, dtype = bm.float64)
        grad[..., 0] = PI * bm.cos(PI * x) * bm.sin(PI * y)
        grad[..., 1] = PI * bm.sin(PI * x) * bm.cos(PI * y)
        return grad

    #源项
    @cartesian
    def source(self, p):

        sol = self.solution(p)
        a = self.diffusion_coefficient(p)
        b = self.reaction_coefficient(p)
        f = bm.zeros(p.shape[:-1], dtype = bm.float64)
        f[:] = 2 * a * PI**2 * sol + b * sol**3
        return f

    #边界条件
    @cartesian                       
    def dirichlet(self, p):
        return self.solution(p)
    
    #非线性项
    def kernel_func_reaction(self, u):
        return u**3
    
    #非线性项的梯度
    def grad_kernel_func_reaction(self, u):
        return 3*u**2
