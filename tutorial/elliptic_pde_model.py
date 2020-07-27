import numpy as np 

from fealpy.decorator import cartesian

class EllipticPDEModel:

    def __init__(self):
        pass

    @cartesian
    def solution(self, p):
        pass

    @cartesian
    def source(self, p):
        pass

    @cartesian
    def gradient(self, p):
        pass

    @cartesian
    def diffusion_coefficient(self, p):
        return np.array([[10.0, -1.0], [-1.0, 2.0]], dtype=np.float64)

    @cartesian
    def convection_coefficient(self, p):
        return np.array([1.0, 1.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x**2 + y**2

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

