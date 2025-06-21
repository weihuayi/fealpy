import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian, barycentric
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl, gradient

class DFPDE2d():
    def __init__(self, u, p, mu, beta, rho):
        """
        @brief: u 是速度场， p 是压力场
        """
        self.mu = mu
        self.beta = beta
        self.rho = rho

        C = CoordSys3D('C')
        x = sym.symbols("x")
        y = sym.symbols("y")

        ux = u.dot(C.i).subs({C.x:x, C.y:y})
        uy = u.dot(C.j).subs({C.x:x, C.y:y})

        Au = (mu + beta*sym.sqrt(ux**2 + uy**2))/rho
        fx = (Au*ux + p.diff(C.x)).subs({C.x:x, C.y:y})
        fy = (Au*uy + p.diff(C.y)).subs({C.x:x, C.y:y})

        p = p.subs({C.x:x, C.y:y})

        self.ux = sym.lambdify(('x', 'y'), ux, "numpy")
        self.uy = sym.lambdify(('x', 'y'), uy, "numpy")

        self.p = sym.lambdify(('x', 'y'), p, "numpy")

        self.fx = sym.lambdify(('x', 'y'), fx, "numpy") 
        self.fy = sym.lambdify(('x', 'y'), fy, "numpy")


    def nonlinear_operator(self, uh):
        mu = self.mu
        beta = self.beta
        rho = self.rho
        @barycentric
        def ret(bcs, index=None):
            """非线性参数：mu + beta*｜u｜"""
            # x : uh(bcs)[..., 0] 
            # y : uh(bcs)[..., 1]
            val = uh(bcs)
            return (mu + beta*np.sqrt(val[:, 0]**2 + val[:, 1]**2))/rho
        return ret

    def nonlinear_operator0(self, uh):
        mu = self.mu
        beta = self.beta
        rho = self.rho
        @barycentric
        def ret(bcs, index=None):
            """非线性参数：mu + beta*｜u｜"""
            # x : uh(bcs)[..., 0] 
            # y : uh(bcs)[..., 1]
            val = uh(bcs)
            return val.swapaxes(-1, -2)*((mu + beta*np.sqrt(val[:, 0]**2 + val[:,
                                                                              1]**2))/rho)[..., None]
        return ret

    @cartesian
    def solutionu(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        ux = self.ux(x, y)
        uy = self.uy(x, y)
        if type(ux) is not np.ndarray:
            ux = np.ones(x.shape, dtype=np.float_)*ux
        if type(uy) is not np.ndarray:
            uy = np.ones(x.shape, dtype=np.float_)*uy
        u = np.concatenate([ux, uy], axis=-1)
        return u 

    @cartesian
    def solutionp(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        p = self.p(x, y)
        if type(p) is not np.ndarray:
            p = np.ones(x.shape, dtype=np.float_)*p
        return p 

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        fx = self.fx(x, y)
        fy = self.fy(x, y)
        if type(fx) is not np.ndarray:
            fx = np.ones(x.shape, dtype=np.float_)*fx
        if type(fy) is not np.ndarray:
            fy = np.ones(x.shape, dtype=np.float_)*fy
        f = np.concatenate([fx, fy], axis=-1)
        return f

    @cartesian
    def neumann(self, p, n):
        """!
        @param p: (..., NE, 2)
        @param n: (NE, 2)
        """
        return np.einsum("...ij,ij->...i", self.solutionu(p), n)

class Data0(DFPDE2d):
    def __init__(self, mu = 1, beta=30, rho=1):
        """
        @brief: 
        @note : \int p dx = 0
        """
        C = CoordSys3D('C')
        u = (C.x+1)**2/4*C.i -(C.x+1)*(C.y+1)/2*C.j
        p = C.x**3 + C.y**3
        #u = (0*C.x + 1)*C.i + (0*C.y + 1)*C.j
        #p = C.x + C.y - 2
        super(Data0, self).__init__(u, p, mu, beta, rho)

    def init_mesh(self, nx=1, ny=1, meshtype='tri'):
        box = [-1, 1, -1, 1]
        mesh = TriangleMesh.from_box(box, nx=nx, ny=ny)
        return mesh


