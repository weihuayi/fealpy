
import numpy as np
from fealpy.mesh import TetrahedronMesh, HexahedronMesh
from fealpy.decorator import cartesian, barycentric
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

class Bubble3dData():
    def __init__(self):
        self.omega = -1

    @cartesian
    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        f = (x**2-x)**2*(y**2-y)**2*(z**2-z)**2
        return np.c_[f, np.sin(x)*f, np.sin(y)*f]

    def mu(self, p):
        return 1

    def epsilon(self, p):
        return 1

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        X = ((4*x - 2)*(x**2 - x)*(4*y - 2)*(y**2 - y)*(z**2 - z)**2*np.sin(x) + (4*x - 2)*(x**2 - x)*(y**2 - y)**2*(4*z - 2)*(z**2 - z)*np.sin(y) - (x**2 - x)**2*(2*y - 1)*(4*y - 2)*(z**2 - z)**2 + (x**2 - x)**2*(4*y - 2)*(y**2 - y)*(z**2 - z)**2*np.cos(x) - (x**2 - x)**2*(y**2 - y)**2*(2*z - 1)*(4*z - 2) - 4*(x**2 - x)**2*(y**2 - y)**2*(z**2 - z) - 4*(x**2 - x)**2*(y**2 - y)*(z**2 - z)**2)
        Y = (-(2*x - 1)*(4*x - 2)*(y**2 - y)**2*(z**2 - z)**2*np.sin(x) + (4*x - 2)*(x**2 - x)*(4*y - 2)*(y**2 - y)*(z**2 - z)**2 - 2*(4*x - 2)*(x**2 - x)*(y**2 - y)**2*(z**2 - z)**2*np.cos(x) + (x**2 - x)**2*(4*y - 2)*(y**2 - y)*(4*z - 2)*(z**2 - z)*np.sin(y) - (x**2 - x)**2*(y**2 - y)**2*(2*z - 1)*(4*z - 2)*np.sin(x) + (x**2 - x)**2*(y**2 - y)**2*(4*z - 2)*(z**2 - z)*np.cos(y) + (x**2 - x)**2*(y**2 - y)**2*(z**2 - z)**2*np.sin(x) - 4*(x**2 - x)**2*(y**2 - y)**2*(z**2 - z)*np.sin(x) - 4*(x**2 - x)*(y**2 - y)**2*(z**2 - z)**2*np.sin(x))
        Z = -(2*x - 1)*(4*x - 2)*(y**2 - y)**2*(z**2 - z)**2*np.sin(y) + (4*x - 2)*(x**2 - x)*(y**2 - y)**2*(4*z - 2)*(z**2 - z) - (x**2 - x)**2*(2*y - 1)*(4*y - 2)*(z**2 - z)**2*np.sin(y) + (x**2 - x)**2*(4*y - 2)*(y**2 - y)*(4*z - 2)*(z**2 - z)*np.sin(x) - 2*(x**2 - x)**2*(4*y - 2)*(y**2 - y)*(z**2 - z)**2*np.cos(y) + (x**2 - x)**2*(y**2 - y)**2*(z**2 - z)**2*np.sin(y) - 4*(x**2 - x)**2*(y**2 - y)*(z**2 - z)**2*np.sin(y) - 4*(x**2 - x)*(y**2 - y)**2*(z**2 - z)**2*np.sin(y)
        return np.c_[X, Y, Z]-self.solution(p)

    def init_mesh(self, n=0):
        box = [0, 1, 0, 1, 0, 1]
        mesh = TetrahedronMesh.from_box(box, nx=n, ny=n, nz=n)
        return mesh

    def dirichlet(self, p):
        return self.solution(p)

    def domain(self):
        box = [0, 1, 0, 1, 0, 1]
        return box 
