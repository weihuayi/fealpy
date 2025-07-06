from fealpy.backend import backend_manager as bm
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
        print('ux:',ux)
        print(sp.simplify(ux))
        uy = sp.diff(u, y)
        print('uy:',uy)
        print(sp.simply(uy))
        uz = sp.diff(u, z)
        print('uz:',uz)
        print(sp.simplify(uz))
        uxx = sp.diff(ux, x)
        print('uxx:',uxx)
        print(sp.simplify(uxx))
        uyy = sp.diff(uy, y)
        print('uyy:',uyy)
        print(sp.simplify(uyy))
        uzz = sp.diff(uz, z)
        print('uzz:',uzz)
        print(sp.simplify(uzz))

        Lu = uxx+uyy+uzz
        print('Lu:', Lu)
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
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        return bm.array(-self.Lu(x, y, z))

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        return bm.array(self.u(x, y, z)) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.ux(x, y,z)
        val[..., 1] = self.uy(x, y,z) 
        val[..., 2] = self.uz(x, y,z)
        return val

    def dirichlet(self, p):
        return self.solution(p)

class DoubleLaplacePDE():
    def __init__(self, u, device=None):
        self.device = device
        self.su = u
        x = sp.symbols("x")
        y = sp.symbols("y")
        z = sp.symbols("z")

        ux = sp.diff(u, x)
        print('ux:',ux)
        uy = sp.diff(u, y)
        print('uy:', uy)
        uz = sp.diff(u, z)
        print('uz:', uz)
        uxx = sp.diff(ux, x)
        print('uxx:', uxx)
        uyy = sp.diff(uy, y)
        print('uyy:', uyy)
        uzz = sp.diff(uz, z)
        print('uzz:', uzz)
        uxy = sp.diff(ux, y)
        print('uxy:', uxy)
        uxz = sp.diff(ux, z)
        print('uxz:', uxz)
        uyz = sp.diff(uy, z)
        print('uyz:', uyz)
        

        Lu = uxx+uyy+uzz
        print('Lu:', Lu)
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2) + sp.diff(Lu, z, 2)
        print('L2u:', L2u)
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
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        return bm.array(self.L2u(x, y, z), device=self.device)

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)

        return bm.array(self.u(x, y, z), device=self.device) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape, dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.ux(x, y, z), device=self.device)
        val[..., 1] = bm.array(self.uy(x, y, z), device=self.device) 
        val[..., 2] = bm.array(self.uz(x, y, z), device=self.device)
        return val

    @cartesian
    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape[:-1]+(6, ), dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.uxx(x, y, z), device=self.device) 
        val[..., 1] = bm.array(self.uxy(x, y, z), device=self.device) 
        val[..., 2] = bm.array(self.uxz(x, y, z), device=self.device)
        val[..., 3] = bm.array(self.uyy(x, y, z), device=self.device) 
        val[..., 4] = bm.array(self.uyz(x, y, z), device=self.device) 
        val[..., 5] = bm.array(self.uzz(x, y, z), device=self.device)
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
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
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
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.ux(x, y, z)
        val[..., 1] = self.uy(x, y, z) 
        val[..., 2] = self.uz(x, y, z)
        return val

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape[:-1]+(6, ), dtype=bm.float64)
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
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape[:-1]+(10, ), dtype=bm.float64)

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


def get_flist(u_sp, device=None): 
    x = sp.symbols("x")
    y = sp.symbols("y")
    z = sp.symbols("z")

    #u_sp = sp.sin(4*x)*sp.cos(5*y)
    #u_sp = x*y
    ux_sp = sp.diff(u_sp, x)
    print('ux_sp:', ux_sp)
    print(sp.simplify(ux_sp))
    uy_sp = sp.diff(u_sp, y)
    print('uy_sp:', uy_sp)
    print(sp.simplify(uy_sp))
    uz_sp = sp.diff(u_sp, z)
    print('uz_sp:', uz_sp)    
    print(sp.simplify(uz_sp))

    uxx_sp = sp.diff(ux_sp, x)
    print('uxx_sp:', uxx_sp)
    print(sp.simplify(uxx_sp))
    uxy_sp = sp.diff(ux_sp, y)
    print('uxy_sp:', uxy_sp)
    print(sp.simplify(uxy_sp))
    uxz_sp = sp.diff(ux_sp, z) 
    print('uxz_sp:', uxz_sp)
    print(sp.simplify(uxz_sp))
    uyy_sp = sp.diff(uy_sp, y)
    print('uyy_sp:', uyy_sp)
    print(sp.simplify(uyy_sp))
    uyz_sp = sp.diff(uy_sp, z)
    print('uyz_sp:', uyz_sp)
    print(sp.simplify(uyz_sp))
    uzz_sp = sp.diff(uz_sp, z)
    print('uzz_sp:', uzz_sp)
    print(sp.simplify(uzz_sp))

    uxxx_sp = sp.diff(uxx_sp, x) # 这个顺序应该怎么写
    print('uxxx_sp:', uxxx_sp)
    uyxx_sp = sp.diff(uxy_sp, x)
    print('uyxx_sp:', uyxx_sp)
    uxxz_sp = sp.diff(uxx_sp, z)
    print('uxxz_sp:', uxxz_sp)
    uyyx_sp = sp.diff(uyy_sp, x)
    print('uyyx_sp:', uyyx_sp)
    uxyz_sp = sp.diff(uxy_sp, z)
    print('uxyz_sp:', uxyz_sp)
    uxzz_sp = sp.diff(uxz_sp, z)
    print('uxzz_sp:', uxzz_sp)
    uyyy_sp = sp.diff(uyy_sp, y)
    print('uyyy_sp:', uyyy_sp)
    uyyz_sp = sp.diff(uyy_sp, z)
    print('uyyz_sp:', uyyz_sp)
    uyzz_sp = sp.diff(uyz_sp, z)
    print('uyzz_sp:', uyzz_sp)
    uzzz_sp = sp.diff(uzz_sp, z)
    print('uzzz_sp:', uzzz_sp)

    uxxxx_sp = sp.diff(uxxx_sp, x)
    print('uxxxx_sp:', uxxxx_sp)
    uyxxx_sp = sp.diff(uyxx_sp, x)
    print('uyxxx_sp:', uyxxx_sp)
    uxxxz_sp = sp.diff(uxxx_sp, z)
    print('uxxxz_sp:', uxxxz_sp)
    uyyxx_sp = sp.diff(uyyx_sp, x)
    print('uyyxx_sp:', uyyxx_sp)
    uxxyz_sp = sp.diff(uxyz_sp, x)
    print('uxxyz_sp:', uxxyz_sp)
    uxxzz_sp = sp.diff(uxxz_sp, z)
    print('uxxzz_sp:', uxxzz_sp)
    uyyyx_sp = sp.diff(uyyy_sp, x)
    print('uyyyx_sp:', uyyyx_sp)
    uxyyz_sp = sp.diff(uxyz_sp, y)
    print('uxyyz_sp:', uxyyz_sp)
    uxyzz_sp = sp.diff(uxyz_sp, z)
    print('uxyzz_sp:', uxyzz_sp)
    uxzzz_sp = sp.diff(uxzz_sp, z)
    print('uxzzz_sp:', uxzzz_sp)
    uyyyy_sp = sp.diff(uyyy_sp, y)
    print('uyyyy_sp:', uyyyy_sp)
    uyyyz_sp = sp.diff(uyyy_sp, z)
    print('uyyyz_sp:', uyyyz_sp)
    uyyzz_sp = sp.diff(uyyz_sp, z)
    print('uyyzz_sp:', uyyzz_sp)
    uyzzz_sp = sp.diff(uyzz_sp, z)
    print('uyzzz_sp:', uyzzz_sp)
    uzzzz_sp = sp.diff(uzzz_sp, z)
    print('uzzzz_sp:', uzzzz_sp)

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

    #f    = lambda node : u(node[..., 0], node[..., 1], node[..., 2])

    @cartesian
    def f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        return bm.array(u(x, y, z), device=device)

    @cartesian
    def grad_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = bm.zeros_like(node)
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        val[..., 0] = bm.array(ux(x, y, z), device=device)
        val[..., 1] = bm.array(uy(x, y, z), device=device)
        val[..., 2] = bm.array(uz(x, y, z), device=device)
        return bm.array(val) 
    @cartesian
    def grad_2_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = bm.zeros(x.shape+(6, ), dtype=bm.float64)
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        val[..., 0] = bm.array(uxx(x, y, z), device=device)
        val[..., 1] = bm.array(uxy(x, y, z), device=device)
        val[..., 2] = bm.array(uxz(x, y, z), device=device)
        val[..., 3] = bm.array(uyy(x, y, z), device=device)
        val[..., 4] = bm.array(uyz(x, y, z), device=device)
        val[..., 5] = bm.array(uzz(x, y, z), device=device)
        return bm.array(val, device=device) 
    def grad_3_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = bm.zeros(x.shape+(10, ), dtype=bm.float64)
        val = bm.to_numpy(val)
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        
        val[..., 0] = bm.array(uxxx(x, y, z), device=device)          
        val[..., 1] = bm.array(uxxy(x, y, z), device=device)
        val[..., 2] = bm.array(uxxz(x, y, z), device=device) 
        val[..., 3] = bm.array(uxyy(x, y, z), device=device) 
        val[..., 4] = bm.array(uxyz(x, y, z), device=device)
        val[..., 5] = bm.array(uxzz(x, y, z), device=device)
        val[..., 6] = bm.array(uyyy(x, y, z), device=device)
        val[..., 7] = bm.array(uyyz(x, y, z), device=device)
        val[..., 8] = bm.array(uyzz(x, y, z), device=device)
        val[..., 9] = bm.array(uzzz(x, y, z), device=device)
        val = bm.array(val, device=device)
        return val 
    def grad_4_f(node):
        x = node[..., 0]
        y = node[..., 1]
        z = node[..., 2]
        val = bm.zeros(x.shape+(15, ), dtype=bm.float64)
        val = bm.to_numpy(val)
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        z = bm.to_numpy(z)
        val[..., 0] =bm.array(uxxxx(x, y, z), device=device)  
        val[..., 1] =bm.array(uyxxx(x, y, z), device=device)
        val[..., 2] =bm.array(uxxxz(x, y, z), device=device)
        val[..., 3] =bm.array(uyyxx(x, y, z), device=device)
        val[..., 4] =bm.array(uxxyz(x, y, z), device=device)
        val[..., 5] =bm.array(uxxzz(x, y, z), device=device)
        val[..., 6] =bm.array(uyyyx(x, y, z), device=device)
        val[..., 7] =bm.array(uxyyz(x, y, z), device=device)
        val[..., 8] =bm.array(uxyzz(x, y, z), device=device)
        val[..., 9] =bm.array(uxzzz(x, y, z), device=device)
        val[..., 10] = bm.array(uyyyy(x, y, z), device=device) 
        val[..., 11] = bm.array(uyyyz(x, y, z), device=device)
        val[..., 12] = bm.array(uyyzz(x, y, z), device=device)
        val[..., 13] = bm.array(uyzzz(x, y, z), device=device)
        val[..., 14] = bm.array(uzzzz(x, y, z), device=device)
        val = bm.array(val,device=device)
        return val 

    flist = [f, grad_f, grad_2_f, grad_3_f, grad_4_f]
    return flist
