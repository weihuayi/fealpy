import numpy as np
import time
import sympy as sp
from fealpy.mesh import TriangleMesh
class NSPDE():
    def __init__(self, u1, u2, p):
        self.box = [0, 1, 0, 1]
        self.eps = 1e-8
        self.mesh = self.set_mesh()

        x, y, t= sp.symbols('x y t')

        gradu1t = sp.diff(u1, t)
        gradu2t = sp.diff(u2, t)
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
        f11 = gradu1t + ulambdau1 + px - delta1
        f22 = gradu2t + ulambdau2 + py - delta2
        #f1 = sp.simplify(f11)
        #f2 = sp.simplify(f22)
        #print(p)
        self.fx = sp.lambdify((x, y, t), f11, "numpy")
        self.fy = sp.lambdify((x, y, t), f22, "numpy")
        self.u1 = sp.lambdify((x, y, t), u1, "numpy")
        self.u2 = sp.lambdify((x, y, t), u2, "numpy")
        self.p = sp.lambdify((x, y, t), p, "numpy")
        

    def domain(self):
        return self.box
    
    def set_mesh(self, n=16):
        box = [0, 1, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        self.mesh = mesh
        return mesh
    
    def velocity(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u1(x, y, t)
        val[..., 1] = self.u2(x, y, t)
        return val
    
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = self.p(x, y, t)
        return val
    
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.fx(x, y, t)         
        val[..., 1] = self.fy(x, y, t)
        return val
    
    def initial_velocity(self, p):
        return self.velocity(p, 0) 

    def is_wall_boundary(self, p):
        eps = self.eps
        return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps) 
    
    def is_u_boundary(self, p):
        eps = self.eps
        return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps) | (np.abs(p[..., 0]) < eps)
    
    def is_right_boundary(self, p):
        return np.abs(p[..., 0] - 1) < self.eps

 
