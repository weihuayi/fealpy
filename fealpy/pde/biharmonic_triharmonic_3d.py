import numpy as np
import sympy as sp
from fealpy.mesh import TetrahedronMesh 
#from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian

class LaplacePDE():
    def __init__(self, u):
        x = sp.symbols("x")
        y = sp.symbols("y")
        z = sp.symbols("z")

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uz = sp.diff(u, z)
        uxx = sp.diff(ux, x)
        uyy = sp.diff(uy, y)
        uzz = sp.diff(uz, z)

        Lu = uxx+uyy+uzz
        self.u = sp.lambdify(('x', 'y', 'z'), u, 'numpy') 

        self.ux = sp.lambdify(('x', 'y', 'z'), ux, 'numpy') 
        self.uy = sp.lambdify(('x', 'y', 'z'), uy, 'numpy') 
        self.uz = sp.lambdify(('x', 'y', 'z'), uz, 'numpy') 
        self.Lu = sp.lambdify(('x', 'y', 'z'), Lu, 'numpy')

    def init_mesh(self, n=1, meshtype='poly'):
        mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 4, 4, 4)
        return mesh

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        return -self.Lu(x, y, z)

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.u(x, y, z) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.ux(x, y)
        val[..., 1] = self.uy(x, y) 
        val[..., 2] = self.uz(x, y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

class DoubleLaplacePDE():
    def __init__(self, u):
        x = sp.symbols("x")
        y = sp.symbols("y")
        z = sp.symbols("z")

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uz = sp.diff(u, z)
        uxx = sp.diff(ux, x)
        uyy = sp.diff(uy, y)
        uzz = sp.diff(uz, z)
        uxy = sp.diff(ux, y)
        uxz = sp.diff(ux, z)
        uyz = sp.diff(uy, z)
        

        Lu = uxx+uyy+uzz
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2) + sp.diff(Lu, z, 2)
        self.u = sp.lambdify(('x', 'y', 'z'), u, 'numpy') 

        self.ux = sp.lambdify(('x', 'y', 'z'), ux, 'numpy') 
        self.uy = sp.lambdify(('x', 'y', 'z'), uy, 'numpy') 
        self.uz = sp.lambdify(('x', 'y', 'z'), uz, 'numpy') 

        self.uxx = sp.lambdify(('x', 'y', 'z'), uxx, 'numpy') 
        self.uxy = sp.lambdify(('x', 'y', 'z'), uxy, 'numpy') 
        self.uxz = sp.lambdify(('x', 'y', 'z'), uxz, 'numpy') 
        self.uyy = sp.lambdify(('x', 'y', 'z'), uyy, 'numpy') 
        self.uyz = sp.lambdify(('x', 'y', 'z'), uyz, 'numpy') 
        self.uzz = sp.lambdify(('x', 'y', 'z'), uzz, 'numpy')

        self.L2u = sp.lambdify(('x', 'y', 'z'), L2u, 'numpy')

    def init_mesh(self, n=1, meshtype='poly'):
        mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 4, 4, 4)
        return mesh

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.L2u(x, y, z)

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.u(x, y, z) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.ux(x, y, z)
        val[..., 1] = self.uy(x, y, z) 
        val[..., 2] = self.uz(x, y, z)
        return val

    @cartesian
    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        val = np.zeros(p.shape[:-1]+(6, ), dtype=np.float64)
        val[..., 0] = self.uxx(x, y, z) 
        val[..., 1] = self.uxy(x, y, z) 
        val[..., 2] = self.uxz(x, y, z)
        val[..., 3] = self.uyy(x, y, z) 
        val[..., 4] = self.uyz(x, y, z) 
        val[..., 5] = self.uzz(x, y, z)
        return val

    def dirichlet(self, p):
        return self.solution(p)

class TripleLaplacePDE():
    def __init__(self, u):
        x = sp.symbols("x")
        y = sp.symbols("y")
        z = sp.symbols("z")

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uz = sp.diff(u, z)

        uxx = sp.diff(ux, x)
        uxy = sp.diff(ux, y)
        uxz = sp.diff(ux, z)
        uyy = sp.diff(uy, y)
        uyz = sp.diff(uy, z)
        uzz = sp.diff(uz, z)
        
        uxxx = sp.diff(uxx, x)
        uxxy = sp.diff(uxx, y)
        uxxz = sp.diff(uxx, z)
        uxyy = sp.diff(uxy, y)
        uxyz = sp.diff(uxy, z)
        uxzz = sp.diff(uxz, z)
        uyyy = sp.diff(uyy, y)
        uyyz = sp.diff(uyy, z)
        uyzz = sp.diff(uyz, z)
        uzzz = sp.diff(uzz, z)

        Lu = uxx+uyy+uzz
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2) + sp.diff(Lu, z, 2)
        L3u = sp.diff(L2u, x, 2) + sp.diff(L2u, y, 2) + sp.diff(L2u, z, 2)

        self.u = sp.lambdify(('x', 'y', 'z'), u, 'numpy') 

        self.ux = sp.lambdify(('x', 'y', 'z'), ux, 'numpy') 
        self.uy = sp.lambdify(('x', 'y', 'z'), uy, 'numpy') 
        self.uz = sp.lambdify(('x', 'y', 'z'), uz, 'numpy') 

        self.uxx = sp.lambdify(('x', 'y', 'z'), uxx, 'numpy') 
        self.uxy = sp.lambdify(('x', 'y', 'z'), uxy, 'numpy') 
        self.uxz = sp.lambdify(('x', 'y', 'z'), uxz, 'numpy') 
        self.uyy = sp.lambdify(('x', 'y', 'z'), uyy, 'numpy') 
        self.uyz = sp.lambdify(('x', 'y', 'z'), uyz, 'numpy') 
        self.uzz = sp.lambdify(('x', 'y', 'z'), uzz, 'numpy')


        self.uxxx = sp.lambdify(('x', 'y', 'z'), uxxx, 'numpy') 
        self.uxxy = sp.lambdify(('x', 'y', 'z'), uxxy, 'numpy') 
        self.uxxz = sp.lambdify(('x', 'y', 'z'), uxxz, 'numpy') 
        #self.uxyx = sp.lambdify(('x', 'y', 'z'), uxyx, 'numpy') 
        self.uxyy = sp.lambdify(('x', 'y', 'z'), uxyy, 'numpy') 
        self.uxyz = sp.lambdify(('x', 'y', 'z'), uxyz, 'numpy') 
        self.uxzz = sp.lambdify(('x', 'y', 'z'), uxzz, 'numpy')
        #self.uyxx = sp.lambdify(('x', 'y', 'z'), uyxx, 'numpy') 
        #self.uyxy = sp.lambdify(('x', 'y', 'z'), uyxy, 'numpy') 
        #self.uyyx = sp.lambdify(('x', 'y', 'z'), uyyx, 'numpy') 
        self.uyyy = sp.lambdify(('x', 'y', 'z'), uyyy, 'numpy') 
        self.uyyz = sp.lambdify(('x', 'y', 'z'), uyyz, 'numpy')
        self.uyzz = sp.lambdify(('x', 'y', 'z'), uyzz, 'numpy')
        self.uzzz = sp.lambdify(('x', 'y', 'z'), uzzz, 'numpy')

        self.L3u = sp.lambdify(('x', 'y', 'z'), L3u, 'numpy')

    def init_mesh(self, n=1, meshtype='poly'):
        mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 4, 4, 4)
        return mesh

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        return -self.L3u(x, y, z) 

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.u(x, y, z) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.ux(x, y, z)
        val[..., 1] = self.uy(x, y, z) 
        val[..., 2] = self.uz(x, y, z)
        return val

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        val = np.zeros(p.shape[:-1]+(6, ), dtype=np.float64)
        val[..., 0] = self.uxx(x, y, z) 
        val[..., 1] = self.uxy(x, y, z) 
        val[..., 2] = self.uxz(x, y, z)
        val[..., 3] = self.uyy(x, y, z) 
        val[..., 4] = self.uyz(x, y, z) 
        val[..., 5] = self.uzz(x, y, z)
        return val

    def grad_3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = np.sin
        pi = np.pi
        cos = np.cos
        val = np.zeros(p.shape[:-1]+(10, ), dtype=np.float64)

        val[..., 0] = self.uxxx(x, y, z)              
        val[..., 1] = self.uxxy(x, y, z)
        val[..., 2] = self.uxxz(x, y, z) 
        val[..., 3] = self.uxyy(x, y, z) 
        val[..., 4] = self.uxyz(x, y, z)
        val[..., 5] = self.uxzz(x, y, z)
        val[..., 6] = self.uyyy(x, y, z)
        val[..., 7] = self.uyyz(x, y, z)
        val[..., 8] = self.uyzz(x, y, z)
        val[..., 9] = self.uzzz(x, y, z)
        return val

    def dirichlet(self, p):
        return self.solution(p)


def get_flist(u_sp): 
    x = sp.symbols("x")
    y = sp.symbols("y")
    z = sp.symbols("z")

    #u_sp = sp.sin(4*x)*sp.cos(5*y)
    #u_sp = x*y
    ux_sp = sp.diff(u_sp, x)
    uy_sp = sp.diff(u_sp, y)
    uz_sp = sp.diff(u_sp, z)
    uxx_sp = sp.diff(ux_sp, x)
    uxy_sp = sp.diff(ux_sp, y)
    uxz_sp = sp.diff(ux_sp, z) 
    uyy_sp = sp.diff(uy_sp, y)
    uyz_sp = sp.diff(uy_sp, z)
    uzz_sp = sp.diff(uz_sp, z)

    uxxx_sp = sp.diff(uxx_sp, x) # 这个顺序应该怎么写
    uyxx_sp = sp.diff(uxy_sp, x)
    uxxz_sp = sp.diff(uxx_sp, z)
    uyyx_sp = sp.diff(uyy_sp, x)
    uxyz_sp = sp.diff(uxy_sp, z)
    uxzz_sp = sp.diff(uxz_sp, z)
    uyyy_sp = sp.diff(uyy_sp, y)
    uyyz_sp = sp.diff(uyy_sp, z)
    uyzz_sp = sp.diff(uyz_sp, z)
    uzzz_sp = sp.diff(uzz_sp, z)

    uxxxx_sp = sp.diff(uxxx_sp, x)
    uyxxx_sp = sp.diff(uyxx_sp, x)
    uxxxz_sp = sp.diff(uxxx_sp, z)
    uyyxx_sp = sp.diff(uyyx_sp, x)
    uxxyz_sp = sp.diff(uxyz_sp, x)
    uxxzz_sp = sp.diff(uxxz_sp, z)
    uyyyx_sp = sp.diff(uyyy_sp, x)
    uxyyz_sp = sp.diff(uxyz_sp, y)
    uxyzz_sp = sp.diff(uxyz_sp, z)
    uxzzz_sp = sp.diff(uxzz_sp, z)
    uyyyy_sp = sp.diff(uyyy_sp, y)
    uyyyz_sp = sp.diff(uyyy_sp, z)
    uyyzz_sp = sp.diff(uyyz_sp, z)
    uyzzz_sp = sp.diff(uyzz_sp, z)
    uzzzz_sp = sp.diff(uzzz_sp, z)

    u     = sp.lambdify(('x', 'y', 'z'), u_sp, 'numpy') 
    ux    = sp.lambdify(('x', 'y', 'z'), ux_sp, 'numpy') 
    uy    = sp.lambdify(('x', 'y', 'z'), uy_sp, 'numpy') 
    uz    = sp.lambdify(('x', 'y', 'z'), uz_sp, 'numpy') 

    uxx   = sp.lambdify(('x', 'y', 'z'), uxx_sp, 'numpy') 
    uxy   = sp.lambdify(('x', 'y', 'z'), uxy_sp, 'numpy') 
    uxz   = sp.lambdify(('x', 'y', 'z'), uxz_sp, 'numpy')
    uyy   = sp.lambdify(('x', 'y', 'z'), uyy_sp, 'numpy') 
    uyz   = sp.lambdify(('x', 'y', 'z'), uyz_sp, 'numpy')
    uzz   = sp.lambdify(('x', 'y', 'z'), uzz_sp, 'numpy')
    
    uxxx  = sp.lambdify(('x', 'y', 'z'), uxxx_sp, 'numpy') 
    uxxy  = sp.lambdify(('x', 'y', 'z'), uyxx_sp, 'numpy') 
    uxxz  = sp.lambdify(('x', 'y', 'z'), uxxz_sp, 'numpy')
    uxyy  = sp.lambdify(('x', 'y', 'z'), uyyx_sp, 'numpy') 
    uxyz  = sp.lambdify(('x', 'y', 'z'), uxyz_sp, 'numpy')
    uxzz  = sp.lambdify(('x', 'y', 'z'), uxzz_sp, 'numpy')
    uyyy  = sp.lambdify(('x', 'y', 'z'), uyyy_sp, 'numpy') 
    uyyz  = sp.lambdify(('x', 'y', 'z'), uyyz_sp, 'numpy')
    uyzz  = sp.lambdify(('x', 'y', 'z'), uyzz_sp, 'numpy')
    uzzz  = sp.lambdify(('x', 'y', 'z'), uzzz_sp, 'numpy')

    uxxxx = sp.lambdify(('x', 'y', 'z'), uxxxx_sp, 'numpy') 
    uyxxx = sp.lambdify(('x', 'y', 'z'), uyxxx_sp, 'numpy') 
    uxxxz = sp.lambdify(('x', 'y', 'z'), uxxxz_sp, 'numpy')
    uyyxx = sp.lambdify(('x', 'y', 'z'), uyyxx_sp, 'numpy') 
    uxxyz = sp.lambdify(('x', 'y', 'z'), uxxyz_sp, 'numpy')
    uxxzz = sp.lambdify(('x', 'y', 'z'), uxxzz_sp, 'numpy')
    uyyyx = sp.lambdify(('x', 'y', 'z'), uyyyx_sp, 'numpy')
    uxyyz = sp.lambdify(('x', 'y', 'z'), uxyyz_sp, 'numpy')
    uxyzz = sp.lambdify(('x', 'y', 'z'), uxyzz_sp, 'numpy')
    uxzzz = sp.lambdify(('x', 'y', 'z'), uxzzz_sp, 'numpy')
    uyyyy = sp.lambdify(('x', 'y', 'z'), uyyyy_sp, 'numpy') 
    uyyyz = sp.lambdify(('x', 'y', 'z'), uyyyz_sp, 'numpy')
    uyyzz = sp.lambdify(('x', 'y', 'z'), uyyzz_sp, 'numpy')
    uyzzz = sp.lambdify(('x', 'y', 'z'), uyzzz_sp, 'numpy')
    uzzzz = sp.lambdify(('x', 'y', 'z'), uzzzz_sp, 'numpy')

    f    = lambda node : u(node[..., 0], node[..., 1], node[..., 2])
    def grad_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = np.zeros_like(node)
        val[..., 0] = ux(x, y, z)
        val[..., 1] = uy(x, y, z)
        val[..., 2] = uz(x, y, z)
        return val 
    def grad_2_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = np.zeros(x.shape+(6, ), dtype=np.float64)
        val[..., 0] = uxx(x, y, z) 
        val[..., 1] = uxy(x, y, z) 
        val[..., 2] = uxz(x, y, z)
        val[..., 3] = uyy(x, y, z) 
        val[..., 4] = uyz(x, y, z) 
        val[..., 5] = uzz(x, y, z)
        return val 
    def grad_3_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = np.zeros(x.shape+(10, ), dtype=np.float64)
        val[..., 0] = uxxx(x, y, z)              
        val[..., 1] = uxxy(x, y, z)
        val[..., 2] = uxxz(x, y, z) 
        val[..., 3] = uxyy(x, y, z) 
        val[..., 4] = uxyz(x, y, z)
        val[..., 5] = uxzz(x, y, z)
        val[..., 6] = uyyy(x, y, z)
        val[..., 7] = uyyz(x, y, z)
        val[..., 8] = uyzz(x, y, z)
        val[..., 9] = uzzz(x, y, z)
        return val 
    def grad_4_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = np.zeros(x.shape+(15, ), dtype=np.float64)
        val[..., 0] = uxxxx(x, y, z)
        val[..., 1] = uyxxx(x, y, z)
        val[..., 2] = uxxxz(x, y, z)
        val[..., 3] = uyyxx(x, y, z)
        val[..., 4] = uxxyz(x, y, z)
        val[..., 5] = uxxzz(x, y, z)
        val[..., 6] = uyyyx(x, y, z)
        val[..., 7] = uxyyz(x, y, z)
        val[..., 8] = uxyzz(x, y, z)
        val[..., 9] = uxzzz(x, y, z)
        val[..., 10] = uyyyy(x, y, z)
        val[..., 11] = uyyyz(x, y, z)
        val[..., 12] = uyyzz(x, y, z)
        val[..., 13] = uyzzz(x, y, z)
        val[..., 14] = uzzzz(x, y, z)
        return val 

    flist = [f, grad_f, grad_2_f, grad_3_f, grad_4_f]
    return flist
