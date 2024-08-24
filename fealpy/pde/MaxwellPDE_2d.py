import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian, barycentric
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

class MaxwellPDE2d():
    def __init__(self, f, eps = 1, k = 1):
        """
        @brief 求解方程 
                curl curl E - eps E = J     Omega
                          n \times E = g     Gamma0
             - curl E - k n \times E = f     Gamma1
        """
        C = CoordSys3D('C')
        x = sym.symbols("x")
        y = sym.symbols("y")

        # 构造 f
        fx = f.dot(C.i).subs({C.x:x, C.y:y})
        fy = f.dot(C.j).subs({C.x:x, C.y:y})

        self.Fx = sym.lambdify(('x', 'y'), fx, "numpy")
        self.Fy = sym.lambdify(('x', 'y'), fy, "numpy")

        # 构造 curl(f)
        cf = curl(f)
        cfz = cf.dot(C.k).subs({C.x:x, C.y:y})
        self.curlF = sym.lambdify(('x', 'y'), cfz, "numpy")
        print('cfz:', cfz)

        # 构造 curl(curl(f))
        ccf = curl(cf)
        ccfx = ccf.dot(C.i).subs({C.x:x, C.y:y})
        ccfy = ccf.dot(C.j).subs({C.x:x, C.y:y})
        self.curlcurlFx = sym.lambdify(('x', 'y'), ccfx, "numpy")
        self.curlcurlFy = sym.lambdify(('x', 'y'), ccfy, "numpy")
        print('ccfx:', ccfx)
        print('ccfy:', ccfy)

        self.eps = eps
        self.k = k

    @cartesian
    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        Fx = self.Fx(x, y)
        Fy = self.Fy(x, y)
        if type(Fx) is not np.ndarray:
            Fx = np.ones(x.shape, dtype=np.float_)*Fx
        if type(Fy) is not np.ndarray:
            Fy = np.ones(x.shape, dtype=np.float_)*Fy
        f = np.c_[Fx, Fy] 
        return f 

    @cartesian
    def curl_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        curlF = self.curlF(x, y)
        if type(curlF) is not np.ndarray:
            curlF = np.ones(x.shape, dtype=np.float_)*curlF
        return curlF

    @cartesian
    def curl_curl_solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        ccFx = self.curlcurlFx(x, y)
        ccFy = self.curlcurlFy(x, y)
        if type(ccFx) is not np.ndarray:
            ccFx = np.ones(x.shape, dtype=np.float_)*ccFx
        if type(ccFy) is not np.ndarray:
            ccFy = np.ones(x.shape, dtype=np.float_)*ccFy
        ccf = np.c_[ccFx, ccFy] 
        return ccf

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        ccFx = self.curlcurlFx(x, y)
        ccFy = self.curlcurlFy(x, y)
        if type(ccFx) is not np.ndarray:
            ccFx = np.ones(x.shape, dtype=np.float_)*ccFx
        if type(ccFy) is not np.ndarray:
            ccFy = np.ones(x.shape, dtype=np.float_)*ccFy
        ccf = np.c_[ccFx, ccFy] 
        return ccf - self.eps * self.solution(p)

    @cartesian
    def dirichlet(self, p, t):
        val = self.solution(p)
        #return np.einsum('...ed, ed->...e', val, t)
        return val

    @cartesian
    def neumann(self, p, n):
        """!
        @param p: (..., N, ldof, 3)
        @param n: (N, 3)
        """
        pass

    def robin(self, p, n):
        """!
        @param p: (..., N, ldof, 3)
        @param n: (N, 3)
        """
        cf = self.curl_solution(p)
        fval = self.solution(p)
        return -cf - np.cross(n[None, :], fval)

    def boundary_type(self, mesh):
        bdface = mesh.ds.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        isLeftBd   = np.abs(f2n[:, 0]+1)<1e-14
        isRightBd  = np.abs(f2n[:, 0]-1)<1e-14
        isBottomBd = np.abs(f2n[:, 1]+1)<1e-14
        isUpBd     = np.abs(f2n[:, 1]-1)<1e-14
        bd = {"left": bdface[isLeftBd], "right": bdface[isRightBd], 
              "up": bdface[isUpBd], "bottom": bdface[isBottomBd], }
        return bd

class SinData(MaxwellPDE2d):
    def __init__(self, eps = 1, k = 1):
        C = CoordSys3D('C')
        #f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j
        #f = sym.sin(C.y)*C.i + sym.sin(C.x)*C.j
        f = C.x*C.y*(1-C.x)*(1-C.y)*C.i + sym.sin(sym.pi*C.x)*sym.sin(sym.pi*C.y)*C.j
        super(SinData, self).__init__(f, eps, k)

    def init_mesh(self, nx=1, ny=1, meshtype='tri'):
        box = [0, 1, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=nx, ny=ny)
        return mesh

    def domain(self):
        return [0, 1, 0, 1]


