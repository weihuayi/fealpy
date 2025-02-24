import sympy as sp
from ..backend import backend_manager as bm
from ..mesh import TriangleMesh
from fealpy.decorator import cartesian

class LaplacePDE():
    """
    -\Delta u = f
    """
    def __init__(self, u):
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.su = u

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uxx = sp.diff(ux, x)
        uyy = sp.diff(uy, y)

        Lu = uxx+uyy
        self.u = sp.lambdify(('x', 'y'), u, 'numpy') 

        self.ux = sp.lambdify(('x', 'y'), ux, 'numpy') 
        self.uy = sp.lambdify(('x', 'y'), uy, 'numpy') 
        self.Lu = sp.lambdify(('x', 'y'), Lu, 'numpy')

    def init_mesh(self, n=1, meshtype='poly'):
        mesh = TriangleMesh.from_box([0, 1, 0, 1], 4, 4)
        return mesh

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        return -self.Lu(x, y)

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return self.u(x, y) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.ux(x, y)
        val[..., 1] = self.uy(x, y) 
        return val

    def dirichlet(self, p):
        return self.solution(p)

class DoubleLaplacePDE():
    """
    \Delta^2 u = f
    """
    def __init__(self, u, device=None):
        self.device = device
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.su = u

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uxx = sp.diff(ux, x)
        uyy = sp.diff(uy, y)
        uxy = sp.diff(ux, y)

        Lu = uxx+uyy
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2)

        self.u = sp.lambdify(('x', 'y'), u, 'numpy') 

        self.ux = sp.lambdify(('x', 'y'), ux, 'numpy') 
        self.uy = sp.lambdify(('x', 'y'), uy, 'numpy') 

        self.uxx = sp.lambdify(('x', 'y'), uxx, 'numpy') 
        self.uyy = sp.lambdify(('x', 'y'), uyy, 'numpy') 
        self.uxy = sp.lambdify(('x', 'y'), uxy, 'numpy') 

        self.L2u = sp.lambdify(('x', 'y'), L2u, 'numpy')

    def init_mesh(self, n=1, meshtype='poly'):
        mesh = TriangleMesh.from_box([0, 1, 0, 1], 4, 4)
        return mesh

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        #sin = bm.sin
        #pi = bm.pi
        #cos = bm.cos
        return bm.array(self.L2u(x_cpu, y_cpu),device=self.device)

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        return bm.array(self.u(x, y), device=self.device)

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        #sin = bm.sin
        #pi = bm.pi
        #cos = bm.cos
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(p.shape, dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.ux(x, y), device=self.device)
        val[..., 1] = bm.array(self.uy(x, y), device=self.device) 
        return val

    @cartesian
    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        #sin = bm.sin
        #pi = bm.pi
        #cos = bm.cos
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(p.shape[:-1]+(3, ), dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.uxx(x, y), device=self.device)
        val[..., 1] = bm.array(self.uxy(x, y), device=self.device)
        val[..., 2] = bm.array(self.uyy(x, y), device=self.device)
        return val

    def dirichlet(self, p):
        return self.solution(p)

class TripleLaplacePDE():
    """
    -\Delta^3 u = f
    """
    def __init__(self, u):
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.su = u

        ux = sp.diff(u, x)
        uy = sp.diff(u, y)
        uxx = sp.diff(ux, x)
        uyy = sp.diff(uy, y)
        uxy = sp.diff(ux, y)

        uxxx = sp.diff(uxx, x)
        uxxy = sp.diff(uxx, y)
        uxyx = sp.diff(uxy, x)
        uxyy = sp.diff(uxy, y)
        uyxx = sp.diff(uxy, x)
        uyxy = sp.diff(uxy, y)
        uyyx = sp.diff(uyy, x)
        uyyy = sp.diff(uyy, y)

        Lu = uxx+uyy
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2)
        L3u = sp.diff(L2u, x, 2) + sp.diff(L2u, y, 2)

        self.u = sp.lambdify(('x', 'y'), u, 'numpy') 

        self.ux = sp.lambdify(('x', 'y'), ux, 'numpy') 
        self.uy = sp.lambdify(('x', 'y'), uy, 'numpy') 

        self.uxx = sp.lambdify(('x', 'y'), uxx, 'numpy') 
        self.uyy = sp.lambdify(('x', 'y'), uyy, 'numpy') 
        self.uxy = sp.lambdify(('x', 'y'), uxy, 'numpy') 

        self.uxxx = sp.lambdify(('x', 'y'), uxxx, 'numpy') 
        self.uxxy = sp.lambdify(('x', 'y'), uxxy, 'numpy') 
        self.uxyx = sp.lambdify(('x', 'y'), uxyx, 'numpy') 
        self.uxyy = sp.lambdify(('x', 'y'), uxyy, 'numpy') 
        self.uyxx = sp.lambdify(('x', 'y'), uyxx, 'numpy') 
        self.uyxy = sp.lambdify(('x', 'y'), uyxy, 'numpy') 
        self.uyyx = sp.lambdify(('x', 'y'), uyyx, 'numpy') 
        self.uyyy = sp.lambdify(('x', 'y'), uyyy, 'numpy') 

        self.L3u = sp.lambdify(('x', 'y'), L3u, 'numpy')

    def init_mesh(self, n=1, meshtype='poly'):
        mesh = TriangleMesh.from_box([0, 1, 0, 1], 4, 4)
        return mesh
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        return -self.L3u(x, y) 

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return self.u(x, y) 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = self.ux(x, y)
        val[..., 1] = self.uy(x, y) 
        return val

    @cartesian
    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape[:-1]+(3, ), dtype=bm.float64)
        val[..., 0] = self.uxx(x, y) 
        val[..., 1] = self.uxy(x, y) 
        val[..., 2] = self.uyy(x, y) 
        return val

    def grad_3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        sin = bm.sin
        pi = bm.pi
        cos = bm.cos
        val = bm.zeros(p.shape[:-1]+(4, ), dtype=bm.float64)
        val[..., 0] = self.uxxx(x, y) 
        val[..., 1] = self.uyxx(x, y)
        val[..., 2] = self.uyyx(x, y)
        val[..., 3] = self.uyyy(x, y)
        return val

    def dirichlet(self, p):
        return self.solution(p)


def get_flist(u_sp, device=None): 
    x = sp.symbols("x")
    y = sp.symbols("y")

    #u_sp = sp.sin(4*x)*sp.cos(5*y)
    #u_sp = x*y
    ux_sp = sp.diff(u_sp, x)
    uy_sp = sp.diff(u_sp, y)
    uxx_sp = sp.diff(ux_sp, x)
    uyx_sp = sp.diff(ux_sp, y)
    uyy_sp = sp.diff(uy_sp, y)
    uxxx_sp = sp.diff(uxx_sp, x)
    uyxx_sp = sp.diff(uyx_sp, x)
    uyyx_sp = sp.diff(uyy_sp, x)
    uyyy_sp = sp.diff(uyy_sp, y)
    uxxxx_sp = sp.diff(uxxx_sp, x)
    uyxxx_sp = sp.diff(uyxx_sp, x)
    uyyxx_sp = sp.diff(uyyx_sp, x)
    uyyyx_sp = sp.diff(uyyy_sp, x)
    uyyyy_sp = sp.diff(uyyy_sp, y)

    u     = sp.lambdify(('x', 'y'), u_sp, 'numpy') 
    ux    = sp.lambdify(('x', 'y'), ux_sp, 'numpy') 
    uy    = sp.lambdify(('x', 'y'), uy_sp, 'numpy') 
    uxx   = sp.lambdify(('x', 'y'), uxx_sp, 'numpy') 
    uyx   = sp.lambdify(('x', 'y'), uyx_sp, 'numpy') 
    uyy   = sp.lambdify(('x', 'y'), uyy_sp, 'numpy') 
    uxxx  = sp.lambdify(('x', 'y'), uxxx_sp, 'numpy') 
    uyxx  = sp.lambdify(('x', 'y'), uyxx_sp, 'numpy') 
    uyyx  = sp.lambdify(('x', 'y'), uyyx_sp, 'numpy') 
    uyyy  = sp.lambdify(('x', 'y'), uyyy_sp, 'numpy') 
    uxxxx = sp.lambdify(('x', 'y'), uxxxx_sp, 'numpy') 
    uyxxx = sp.lambdify(('x', 'y'), uyxxx_sp, 'numpy') 
    uyyxx = sp.lambdify(('x', 'y'), uyyxx_sp, 'numpy') 
    uyyyx = sp.lambdify(('x', 'y'), uyyyx_sp, 'numpy') 
    uyyyy = sp.lambdify(('x', 'y'), uyyyy_sp, 'numpy') 

    #f    = lambda node : u(node[..., 0], node[..., 1])
    def f(node):
        x = node[..., 0]
        y = node[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.array(u(x_cpu, y_cpu), device=device)
    def grad_f(node):
        x = node[..., 0]
        y = node[..., 1]
        val = bm.zeros_like(node, device=device)
        x_cpu = bm.to_numpy(x) 
        y_cpu = bm.to_numpy(y)
        val[..., 0] = bm.array(ux(x_cpu, y_cpu), device=device)
        val[..., 1] = bm.array(uy(x_cpu, y_cpu), device=device)
        return val 
    def grad_2_f(node):
        x = node[..., 0]
        y = node[..., 1]
        val = bm.zeros(x.shape+(3, ), dtype=bm.float64, device=device)
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val[..., 0] = bm.array(uxx(x, y), device=device)
        val[..., 1] = bm.array(uyx(x, y), device=device)
        val[..., 2] = bm.array(uyy(x, y), device=device)
        return val 
    def grad_3_f(node):
        x = node[..., 0]
        y = node[..., 1]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(x.shape+(4, ), dtype=bm.float64, device=device)
        val[..., 0] = bm.array(uxxx(x, y), device=device)
        val[..., 1] = bm.array(uyxx(x, y), device=device)
        val[..., 2] = bm.array(uyyx(x, y), device=device)
        val[..., 3] = bm.array(uyyy(x, y), device=device)
        return val 
    def grad_4_f(node):
        x = node[..., 0]
        y = node[..., 1]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(x.shape+(5, ), dtype=bm.float64, device=device)
        val[..., 0] = bm.array(uxxxx(x, y), device=device)
        val[..., 1] = bm.array(uyxxx(x, y), device=device)
        val[..., 2] = bm.array(uyyxx(x, y), device=device)
        val[..., 3] = bm.array(uyyyx(x, y), device=device)
        val[..., 4] = bm.array(uyyyy(x, y), device=device)
        return val 

    flist = [f, grad_f, grad_2_f, grad_3_f, grad_4_f]
    return flist
