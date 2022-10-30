import numpy as np
from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.decorator import cartesian, barycentric
import sympy as sym
from sympy.vector import CoordSys3D, Del, curl

class MaxwellPDE():
    def __init__(self, f):
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

    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        Fx = self.Fx(x, y, z)
        Fy = self.Fy(x, y, z)
        Fz = self.Fz(x, y, z)
        if type(Fx) is not np.ndarray:
            Fx = np.ones(x.shape, dtype=np.float_)*Fx
        if type(Fy) is not np.ndarray:
            Fy = np.ones(x.shape, dtype=np.float_)*Fy
        if type(Fz) is not np.ndarray:
            Fz = np.ones(x.shape, dtype=np.float_)*Fz
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
            cFx = np.ones(x.shape, dtype=np.float_)*cFx
        if type(cFy) is not np.ndarray:
            cFy = np.ones(x.shape, dtype=np.float_)*cFy
        if type(cFz) is not np.ndarray:
            cFz = np.ones(x.shape, dtype=np.float_)*cFz
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
            ccFx = np.ones(x.shape, dtype=np.float_)*ccFx
        if type(ccFy) is not np.ndarray:
            ccFy = np.ones(x.shape, dtype=np.float_)*ccFy
        if type(ccFz) is not np.ndarray:
            ccFz = np.ones(x.shape, dtype=np.float_)*ccFz
        ccf = np.c_[ccFx, ccFy, ccFz] 
        return ccf - self.solution(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """!
        @param p: (..., N, ldof, 3)
        @param n: (N, 3)
        """
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        cFx = self.curlFx(x, y, z)
        cFy = self.curlFy(x, y, z)
        cFz = self.curlFz(x, y, z)
        if type(cFx) is not np.ndarray:
            cFx = np.ones(x.shape, dtype=np.float_)*cFx
        if type(cFy) is not np.ndarray:
            cFy = np.ones(x.shape, dtype=np.float_)*cFy
        if type(cFz) is not np.ndarray:
            cFz = np.ones(x.shape, dtype=np.float_)*cFz
        cf = np.c_[cFx, cFy, cFz] #(..., NC, ldof, 3)
        return np.cross(n[None, :], cf)

    def boundary_type(self, mesh):
        bdface = mesh.ds.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        neumannbd = np.abs(f2n[:, 2])<0.9
        bd = {"neumann": bdface[neumannbd], "dirichlet": bdface[~neumannbd]}
        return bd

class SinData(MaxwellPDE):
    def __init__(self):
        C = CoordSys3D('C')
        #f = 1*C.i + sym.sin(sym.pi*C.x)*C.j + sym.sin(sym.pi*C.z)*C.k 
        #f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j + C.x*sym.sin(sym.pi*C.z)*C.k 
        #f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j + C.z*C.k 
        f = sym.sin(sym.pi*C.y)*C.i + sym.sin(sym.pi*C.x)*C.j + sym.sin(sym.pi*C.z)*C.k 
        #f = sym.sin(sym.pi*C.x)*C.i + sym.sin(sym.pi*C.y)*C.j + sym.sin(sym.pi*C.z)*C.k 
        #f = C.x**3*C.i + C.y**3*C.j + C.z**3*C.k
        #f = C.y*C.i + 2*C.x*C.j + C.z*C.k

        #f = (C.x**2-C.x)**2*(C.y**2-C.y)**2*(C.z**2-C.z)**2
        #f = f*C.i + sym.sin(C.x)*f*C.j + sym.sin(C.y)*f*C.k
        super(SinData, self).__init__(f)

    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = MeshFactory.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
        return mesh

class BubbleData(MaxwellPDE):
    def __init__(self):
        C = CoordSys3D('C')
        f = (C.x**2-C.x)*(C.y**2-C.y)*(C.z**2-C.z)
        u = f*C.i + sym.sin(C.x)*f*C.j + sym.sin(C.y)*f*C.k
        super(BubbleData, self).__init__(u)

    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = MeshFactory.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
        return mesh


class XXX3dData():
    def __init__(self, n=2):
        self.n = n
        self.omega = -1

    @cartesian
    def solution(self, p):
        n = self.n
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        #z[:] = 100
        return np.c_[y**n, x**n, z]

    def mu(self, p):
        return 1

    def epsilon(self, p):
        return 1

    @cartesian
    def source(self, p):
        n = self.n
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        z[:] = 0
        return -n*(n-1)*np.c_[y**(n-2), x**(n-2), z]-self.solution(p)

    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = MeshFactory.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
        return mesh

    def dirichlet(self, p):
        return self.solution(p)

class XXX3dData0():
    def __init__(self, n=2):
        self.n = n
        self.omega = -1

    @cartesian
    def solution(self, p):
        n = self.n
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        return np.c_[x**n, y**n, z**n]

    def mu(self, p):
        return 1

    def epsilon(self, p):
        return 1

    @cartesian
    def source(self, p):
        n = self.n
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        z[:] = 0
        return -self.solution(p)

    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = MeshFactory.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
        return mesh

    def dirichlet(self, p):
        return self.solution(p)


class Sin3dData():
    def __init__(self):
        self.omega = -1

    @cartesian
    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        z[:] = 0
        return np.c_[np.sin(np.pi*y), np.sin(np.pi*x), z]

    def mu(self, p):
        return 1

    def epsilon(self, p):
        return 1

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        z[:] = 0
        return np.c_[np.pi**2*np.sin(np.pi*y), np.pi**2*np.sin(np.pi*x), z]-self.solution(p)

    def init_mesh(self, n=1):
        box = [0, 1, 0, 1, 0, 1]
        mesh = MeshFactory.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
        return mesh

    def dirichlet(self, p):
        return self.solution(p)

    def neumann(self, p, n):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        z[:] = 0
        val = np.c_[z, z, np.pi*(np.cos(np.pi*x)-np.cos(np.pi*y))]
        return np.cross(n)

    def boundary_type(self, mesh):
        bdface = mesh.boundary_face_index()
        f2n = mesh.face_normal()[bdface]
        neumannbd = np.abs(f2n[:, 2])>0.9
        bd = {"neumann": bdface[neumannbd], "dirichlet": bdface[~neumannbd]}
        return bd

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
        mesh = MeshFactory.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
        return mesh

    def dirichlet(self, p):
        return self.solution(p)
