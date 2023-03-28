import numpy as np
from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.decorator import cartesian, barycentric
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

class MaxwellPDE2d():
    def __init__(self, f):
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

        # 构造 curl(curl(f))
        ccf = curl(cf)
        ccfx = ccf.dot(C.i).subs({C.x:x, C.y:y})
        ccfy = ccf.dot(C.j).subs({C.x:x, C.y:y})
        self.curlcurlFx = sym.lambdify(('x', 'y'), ccfx, "numpy")
        self.curlcurlFy = sym.lambdify(('x', 'y'), ccfy, "numpy")

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
        return ccf - self.solution(p)

    @cartesian
    def dirichlet(self, p, t):
        val = self.solution(p)
        return np.einsum('...ed, ed->...e', val, t)

    @cartesian
    def neumann(self, p, n):
        """!
        @param p: (..., N, ldof, 3)
        @param n: (N, 3)
        """
        x = p[..., 0, None]
        y = p[..., 1, None]
        cFx = self.curlFx(x, y)
        cFy = self.curlFy(x, y)
        if type(cFx) is not np.ndarray:
            cFx = np.ones(x.shape, dtype=np.float_)*cFx
        if type(cFy) is not np.ndarray:
            cFy = np.ones(x.shape, dtype=np.float_)*cFy
        cf = np.c_[cFx, cFy] #(..., NC, ldof, 3)
        return np.cross(n[None, :], cf)

    def boundary_type(self, mesh):
        bdface = mesh.ds.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        neumannbd = np.abs(f2n[:, 2])<0.9
        bd = {"neumann": bdface[neumannbd], "dirichlet": bdface[~neumannbd]}
        return bd

class SinData(MaxwellPDE2d):
    def __init__(self):
        C = CoordSys3D('C')
        f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j
        #f = sym.sin(C.y)*C.i + sym.sin(C.x)*C.j
        #f = C.x*C.y*(1-C.x)*(1-C.y)*C.i + sym.sin(sym.pi*C.x)*sym.sin(sym.pi*C.y)*C.j
        super(SinData, self).__init__(f)

    def init_mesh(self, nx=1, ny=1, meshtype='tri'):
        box = [0, 1, 0, 1]
        mesh = MeshFactory.boxmesh2d(box, nx=nx, ny=ny, meshtype=meshtype)
        return mesh


