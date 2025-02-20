
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

from ..mesh import TetrahedronMesh, HexahedronMesh
from ..backend import backend_manager as bm
from ..decorator import cartesian, barycentric


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
    
    @cartesian
    def curl_solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        f1 = bm.cos(y)*(x**2 - x)**2 * (y**2 - y)**2*(z**2 - z)**2 + bm.sin(y)*(x**2 - x)**2*(y**2 - y)*2*(2*y-1)*(z**2 - z)**2 - bm.sin(x)*(x**2 - x)**2*(y**2 - y)**2*(z**2 - z)* 2 * (2*z-1)
        f2 = (x**2 - x)**2 * (y**2 - y)**2 * (z**2 - z) * 2 * (2*z-1) - bm.sin(y)*2*(x**2 - x)*(2*x-1)*(y**2 - y) ** 2 *(z**2 - z)**2
        f3 =  bm.cos(x)*(x**2 - x)**2 *(y**2 - y)**2 *(z**2 - z)**2 + bm.sin(x)*(x**2 - x)*2*(2*x-1)*(y**2 - y)**2 *(z**2 - z)**2 - (x**2 - x)**2 *(y**2 - y)* 2 *(2*y-1)*(z**2 - z)**2
        return bm.concatenate([f1, f2, f3], axis=-1)
    
    def k(self,p):    
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


import numpy as np
from ..mesh import TetrahedronMesh, HexahedronMesh
from ..decorator import cartesian, barycentric
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

class MaxwellPDE():
    def __init__(self, f, beta=1, k=1):
        """
        @brief 求解方程 
                             curl curl E - beta E = J     Omega
                                       n \times E = g     Gamma0
                 n \times (curl E + k n \times E) = f     Gamma1
        """
        C = CoordSys3D('C')
        x = sym.symbols("x")
        y = sym.symbols("y")
        z = sym.symbols("z")

        # 构造 f
        fx = f.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
        fy = f.dot(C.j).subs({C.x:x, C.y:y, C.z:z})
        fz = f.dot(C.k).subs({C.x:x, C.y:y, C.z:z})

        self.Fx = sym.lambdify(('x', 'y', 'z'), fx, "numpy")
        self.Fy = sym.lambdify(('x', 'y', 'z'), fy, "numpy")
        self.Fz = sym.lambdify(('x', 'y', 'z'), fz, "numpy")

        # 构造 curl(f)
        cf = curl(f)
        cfx = cf.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
        cfy = cf.dot(C.j).subs({C.x:x, C.y:y, C.z:z})
        cfz = cf.dot(C.k).subs({C.x:x, C.y:y, C.z:z})
        self.curlFx = sym.lambdify(('x', 'y', 'z'), cfx, "numpy")
        self.curlFy = sym.lambdify(('x', 'y', 'z'), cfy, "numpy")
        self.curlFz = sym.lambdify(('x', 'y', 'z'), cfz, "numpy")

        # 构造 curl(curl(f))
        ccf = curl(cf)
        ccfx = ccf.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
        ccfy = ccf.dot(C.j).subs({C.x:x, C.y:y, C.z:z})
        ccfz = ccf.dot(C.k).subs({C.x:x, C.y:y, C.z:z})
        self.curlcurlFx = sym.lambdify(('x', 'y', 'z'), ccfx, "numpy")
        self.curlcurlFy = sym.lambdify(('x', 'y', 'z'), ccfy, "numpy")
        self.curlcurlFz = sym.lambdify(('x', 'y', 'z'), ccfz, "numpy")

        self.beta = beta
        self.k = k

    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        Fx = self.Fx(x, y, z)
        Fy = self.Fy(x, y, z)
        Fz = self.Fz(x, y, z)
        if type(Fx) is not np.ndarray:
            Fx = np.ones(x.shape, dtype=np.float64)*Fx
        if type(Fy) is not np.ndarray:
            Fy = np.ones(x.shape, dtype=np.float64)*Fy
        if type(Fz) is not np.ndarray:
            Fz = np.ones(x.shape, dtype=np.float64)*Fz
        f = np.c_[Fx, Fy, Fz] 
        return f 

    @cartesian
    def curl_solution(self, p):
        """!
        @param p: (..., N, ldof, 3)
        """
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        cFx = self.curlFx(x, y, z)
        cFy = self.curlFy(x, y, z)
        cFz = self.curlFz(x, y, z)
        if type(cFx) is not np.ndarray:
            cFx = np.ones(x.shape, dtype=np.float64)*cFx
        if type(cFy) is not np.ndarray:
            cFy = np.ones(x.shape, dtype=np.float64)*cFy
        if type(cFz) is not np.ndarray:
            cFz = np.ones(x.shape, dtype=np.float64)*cFz
        cf = np.c_[cFx, cFy, cFz] #(..., NC, ldof, 3)
        return cf 

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        ccFx = self.curlcurlFx(x, y, z)
        ccFy = self.curlcurlFy(x, y, z)
        ccFz = self.curlcurlFz(x, y, z)
        if type(ccFx) is not np.ndarray:
            ccFx = np.ones(x.shape, dtype=np.float64)*ccFx
        if type(ccFy) is not np.ndarray:
            ccFy = np.ones(x.shape, dtype=np.float64)*ccFy
        if type(ccFz) is not np.ndarray:
            ccFz = np.ones(x.shape, dtype=np.float64)*ccFz
        ccf = np.c_[ccFx, ccFy, ccFz] 
        return ccf - self.beta*self.solution(p)

    @cartesian
    def dirichlet(self, p):
        val = self.solution(p)
        return val
    
    @cartesian
    def neumann(self, p,n):
        """
        @note p : (NF, NQ, 3)
              n : (NF, 3)
              self.curl_solution(p) : (NF, NQ, 3)
        """     
        return  bm.cross(n[:, None,:], self.curl_solution(p))


class BubbleData(MaxwellPDE):
    def __init__(self):
        C = CoordSys3D('C')
        f = (C.x**2-C.x)*(C.y**2-C.y)*(C.z**2-C.z)
        u = f*C.i + sym.sin(np.pi*C.x)*f*C.j + sym.sin(np.pi*C.y)*f*C.k
        # u =  2*C.i
        super(BubbleData, self).__init__(u)

    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = TetrahedronMesh.from_box(box, nx=n, ny=n, nz=n)
        return mesh
    
    def domain(self):
        box = [0, 1/2, 0, 1/2, 0, 1/2]
        #box = [0, 1, 0, 1, 0, 1]
        return box
     
class BubbleData3d():

    @cartesian
    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        f = (x**2-x)*(y**2-y)*(z**2-z)
        return bm.concatenate([f, bm.sin(bm.pi*x)*f, bm.sin(bm.pi*y)*f], axis=-1)
    
    @cartesian
    def curl_solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        sin = bm.sin
        cos = bm.cos
        pi = bm.pi
        f1 = ((x**2 - x)*(2*y - 1)*(z**2 - z)*sin(y*pi) - (x**2 - x)*(y**2 - y)*(2*z - 1)*sin(x*pi) + pi*(x**2 - x)*(y**2 - y)*(z**2 - z)*cos(y*pi))
        f2 =  (-(2*x - 1)*(y**2 - y)*(z**2 - z)*sin(y*pi) + (x**2 - x)*(y**2 - y)*(2*z - 1))
        f3 =  ((2*x - 1)*(y**2 - y)*(z**2 - z)*sin(x*pi) - (x**2 - x)*(2*y - 1)*(z**2 - z) + pi*(x**2 - x)*(y**2 - y)*(z**2 - z)*cos(x*pi))

        return bm.concatenate([f1, f2, f3], axis=-1)

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        sin = bm.sin
        cos = bm.cos
        pi = bm.pi
        f1 = (-(1 - 2*x)*(y**2 - y)*(2*z - 1)*sin(y*pi) + (2*x - 1)*(2*y - 1)*(z**2 - z)*sin(x*pi) + (-2*x**2 + 2*x)*(z**2 - z) + pi*(x**2 - x)*(2*y - 1)*(z**2 - z)*cos(x*pi) - (x**2 - x)*(y**2 - y)*(z**2 - z) - (2*x**2 - 2*x)*(y**2 - y))
        f2 = (-(1 - 2*y)*(2*x - 1)*(z**2 - z) - 2*pi*(2*x - 1)*(y**2 - y)*(z**2 - z)*cos(x*pi) + (-2*x**2 + 2*x)*(y**2 - y)*sin(x*pi) + (x**2 - x)*(2*y - 1)*(2*z - 1)*sin(y*pi) + pi*(x**2 - x)*(y**2 - y)*(2*z - 1)*cos(y*pi) - (x**2 - x)*(y**2 - y)*(z**2 - z)*sin(x*pi) + pi**2*(x**2 - x)*(y**2 - y)*(z**2 - z)*sin(x*pi) - (2*y**2 - 2*y)*(z**2 - z)*sin(x*pi))
        f3 = (-(1 - 2*z)*(x**2 - x)*(2*y - 1)*sin(x*pi) + (2*x - 1)*(y**2 - y)*(2*z - 1) - 2*pi*(x**2 - x)*(2*y - 1)*(z**2 - z)*cos(y*pi) - (x**2 - x)*(y**2 - y)*(z**2 - z)*sin(y*pi) + pi**2*(x**2 - x)*(y**2 - y)*(z**2 - z)*sin(y*pi) - (2*x**2 - 2*x)*(z**2 - z)*sin(y*pi) + (-2*y**2 + 2*y)*(z**2 - z)*sin(y*pi))
        return bm.concatenate([f1, f2, f3], axis=-1)
    
    @cartesian
    def dirichlet(self, p,n):
        val1 = self.solution(p)
        val = bm.cross(val1,n)
        return val
    
    @cartesian
    def neumann(self, p,n):
        """
        @note p : (NF, NQ, 3)
              n : (NF, 3)
              self.curl_solution(p) : (NF, NQ, 3)
        """     
        return  bm.cross(n[:, None,:], self.curl_solution(p))
    
    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = TetrahedronMesh.from_box(box, nx=n, ny=n, nz=n)
        return mesh
    
    def domain(self):
        box = [0, 1/2, 0, 1/2, 0, 1/2]
        #box = [0, 1, 0, 1, 0, 1]
        return box    





