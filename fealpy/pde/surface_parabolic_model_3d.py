import numpy as np

class SinSinSinExpData:
    def __init__(self):
        self.diffusionCoefficient = 1/16

    def init_value(self,p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        x = p[...,: , 0]
        y = p[...,: , 1]
        z = p[...,: , 2]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        u = sin(pi*x)*sin(pi*y)*sin(pi*z)*np.exp(t)
        return u 

    def diffusion_coefficient(self):
        return self.diffusionCoefficient

    def source(self, p, t):        
        x = p[...,: , 0]
        y = p[...,: , 1]
        z = p[...,: , 2]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        K = self.diffusionCoefficient
        rhs = (1+K*pi**2)*sin(pi*x)*sin(pi*y)*sin(pi*z)*np.exp(t)
        return rhs




