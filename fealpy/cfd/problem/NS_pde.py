import numpy as np
import time
import sympy as sp
class StokesPDE():
    def __init__(self, u1, u2, p):
        self.box = [0, 1, 0, 1]
        self.eps = 1e-8
        x, y = sp.symbols('x y')

        gradu1x = sp.diff(u1, x)
        gradu1y = sp.diff(u1, y)
        gradu2x = sp.diff(u2, x)
        gradu2y = sp.diff(u2, y)
        
        assert sp.simplify(gradu1x+gradu2y) == 0
        delta1 = sp.diff(gradu1x, x) + sp.diff(gradu1y, y)
        delta2 = sp.diff(gradu2x, x) + sp.diff(gradu2y, y)
        ulambdau1 = u1 * gradu1x + u2 * gradu1y
        ulambdau2 = u1 * gradu2x + u2 * gradu2y
        px = sp.diff(p, x)
        py = sp.diff(p, y)
        f11 = ulambdau1 + px - delta1
        f22 = ulambdau2 + py - delta2
        self.fx = sp.lambdify((x, y), f11, "numpy")
        self.fy = sp.lambdify((x, y), f22, "numpy")
        self.u1 = sp.lambdify((x, y), u1, "numpy")
        self.u2 = sp.lambdify((x, y), u2, "numpy")
        self.p = sp.lambdify((x, y), p, "numpy")

    def domain(self):
        return self.box
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u1(x, y)
        val[..., 1] = self.u2(x, y)
        return val
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = self.p(x,y)
        return val
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.fx(x,y)         
        val[..., 1] = self.fy(x,y)
        return val
    def initial_velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        value = np.zeros(p.shape, dtype=np.float64)
        value[..., 0] = 1
        value[..., 1] = 1
        return value
    def is_wall_boundary(self, p):
        eps = self.eps
        return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps) 

    def is_left_right_boundary(self, p):
        eps = self.eps
        return (np.abs(p[..., 0]) < eps) | (np.abs(p[..., 0] - 1.0) < eps) 

    def is_u_boundary(self, p):
        eps = self.eps
        return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps) | (np.abs(p[..., 0]) < eps)

    def is_right_boundary(self, p):
        return np.abs(p[..., 0] - 1) < self.eps

 

