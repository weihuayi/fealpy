
from fealpy.mesh import TetrahedronMesh, HexahedronMesh
from fealpy.experimental.backend import backend_manager as bm
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
        return bm.concatenate([f, bm.sin(x)*f, bm.sin(y)*f], axis=-1)

    def mu(self, p):
        return 1

    def epsilon(self, p):
        return 1

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        X = ((4*x - 2)*(x**2 - x)*(4*y - 2)*(y**2 - y)*(z**2 - z)**2*bm.sin(x) + (4*x - 2)*(x**2 - x)*(y**2 - y)**2*(4*z - 2)*(z**2 - z)*bm.sin(y) - (x**2 - x)**2*(2*y - 1)*(4*y - 2)*(z**2 - z)**2 + (x**2 - x)**2*(4*y - 2)*(y**2 - y)*(z**2 - z)**2*bm.cos(x) - (x**2 - x)**2*(y**2 - y)**2*(2*z - 1)*(4*z - 2) - 4*(x**2 - x)**2*(y**2 - y)**2*(z**2 - z) - 4*(x**2 - x)**2*(y**2 - y)*(z**2 - z)**2)
        Y = (-(2*x - 1)*(4*x - 2)*(y**2 - y)**2*(z**2 - z)**2*bm.sin(x) + (4*x - 2)*(x**2 - x)*(4*y - 2)*(y**2 - y)*(z**2 - z)**2 - 2*(4*x - 2)*(x**2 - x)*(y**2 - y)**2*(z**2 - z)**2*bm.cos(x) + (x**2 - x)**2*(4*y - 2)*(y**2 - y)*(4*z - 2)*(z**2 - z)*bm.sin(y) - (x**2 - x)**2*(y**2 - y)**2*(2*z - 1)*(4*z - 2)*bm.sin(x) + (x**2 - x)**2*(y**2 - y)**2*(4*z - 2)*(z**2 - z)*bm.cos(y) + (x**2 - x)**2*(y**2 - y)**2*(z**2 - z)**2*bm.sin(x) - 4*(x**2 - x)**2*(y**2 - y)**2*(z**2 - z)*bm.sin(x) - 4*(x**2 - x)*(y**2 - y)**2*(z**2 - z)**2*bm.sin(x))
        Z = -(2*x - 1)*(4*x - 2)*(y**2 - y)**2*(z**2 - z)**2*bm.sin(y) + (4*x - 2)*(x**2 - x)*(y**2 - y)**2*(4*z - 2)*(z**2 - z) - (x**2 - x)**2*(2*y - 1)*(4*y - 2)*(z**2 - z)**2*bm.sin(y) + (x**2 - x)**2*(4*y - 2)*(y**2 - y)*(4*z - 2)*(z**2 - z)*bm.sin(x) - 2*(x**2 - x)**2*(4*y - 2)*(y**2 - y)*(z**2 - z)**2*bm.cos(y) + (x**2 - x)**2*(y**2 - y)**2*(z**2 - z)**2*bm.sin(y) - 4*(x**2 - x)**2*(y**2 - y)*(z**2 - z)**2*bm.sin(y) - 4*(x**2 - x)*(y**2 - y)**2*(z**2 - z)**2*bm.sin(y)
        return bm.concatenate([X, Y, Z], axis=-1)-self.solution(p)

    def init_mesh(self, n=0):
        box = [0, 1, 0, 1, 0, 1]
        mesh = TetrahedronMesh.from_box(box, nx=n, ny=n, nz=n)
        return mesh

    def dirichlet(self, p):
        return self.solution(p)

    def domain(self):
        box = [0, 1, 0, 1, 0, 1]
        return box 
