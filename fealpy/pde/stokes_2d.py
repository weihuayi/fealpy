import sympy as sp
from ..backend import backend_manager as bm
from ..mesh import TriangleMesh
from fealpy.decorator import cartesian

class StokesStreamPDE():
    """
    \Delta^2 u = f
    """
    def __init__(self, phi, p, viscosity, device=None):
        self.device = device
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.viscosity = viscosity
        print("viscosity:", viscosity)
        self.sphi = phi
        self.p = p
        print('p',p)
        px = sp.diff(p, x)
        py = sp.diff(p, y)
        self.px = sp.lambdify(('x', 'y'), px, 'numpy')
        self.py = sp.lambdify(('x', 'y'), py, 'numpy')

        phix = sp.diff(phi, x)
        #print('phix',phix)
        #print(sp.simplify(phix))
        phiy = sp.diff(phi, y)
        #print('phiy',phiy)
        #print(sp.simplify(phiy))
        #print(sp.trigsimp(phiy))
        phixx = sp.diff(phix, x)
        #print('phixx',phixx)
        #print(sp.simplify(phixx))
        #print(sp.trigsimp(phixx))
        phiyy = sp.diff(phiy, y)
        #print('phiyy',phiyy)
        #print(sp.simplify(phiyy))
        #print(sp.trigsimp(phiyy))
        phixy = sp.diff(phix, y)
        #print('phixy',phixy)
        #print(sp.simplify(phixy))
        #print(sp.trigsimp(phixy))

        Lphi = phixx+phiyy
        #print("Lphi:", Lphi)
        #print(sp.simplify(Lphi))
        #print(sp.trigsimp(Lphi))
        L2phi = sp.diff(Lphi, x, 2) + sp.diff(Lphi, y, 2)
        #print("L2phi:", L2phi)
        #print(sp.simplify(L2phi))
        #print(sp.trigsimp(L2phi))

        u1 = phiy
        u2 = -phix
        print('u1:', u1)
        print('u2:', u2)
        print("phi:", phi)

        u1x = phixy
        u1y = phiyy
        u2x = -phixx
        u2y = -phixy
        assert (u1x+u2y == 0)

        Lu1 = sp.diff(u1x,x)+sp.diff(u1y,y)
        Lu2 = sp.diff(u2x,x)+sp.diff(u2y,y)

        f1 = -viscosity*Lu1 + px 
        f2 = -viscosity*Lu2 + py
        f1x = sp.diff(f1, x)
        f1y = sp.diff(f1, y)
        f2x = sp.diff(f2, x)
        f2y = sp.diff(f2, y)
        fphi = sp.diff(f2, x) - sp.diff(f1, y)
        fp = -f1x - f2y
        

        assert (sp.simplify(fphi) == sp.simplify(viscosity*L2phi))

        self.phi = sp.lambdify(('x', 'y'), phi, 'numpy') 
        self.p = sp.lambdify(('x', 'y'), p, 'numpy') 

        self.phix = sp.lambdify(('x', 'y'), phix, 'numpy') 
        self.phiy = sp.lambdify(('x', 'y'), phiy, 'numpy') 
        self.u1 = sp.lambdify(('x', 'y'), u1, 'numpy')
        self.u2 = sp.lambdify(('x', 'y'), u2, 'numpy')
        self.u1x = sp.lambdify(('x', 'y'), u1x, 'numpy')
        self.u1y = sp.lambdify(('x', 'y'), u1y, 'numpy')
        self.u2x = sp.lambdify(('x', 'y'), u2x, 'numpy')
        self.u2y = sp.lambdify(('x', 'y'), u2y, 'numpy')
        self.Lu1 = sp.lambdify(('x', 'y'), Lu1, 'numpy')
        self.Lu2 = sp.lambdify(('x', 'y'), Lu2, 'numpy')
        self.f1 = sp.lambdify(('x', 'y'), f1, 'numpy')
        self.f2 = sp.lambdify(('x', 'y'), f2, 'numpy')
        self.fp = sp.lambdify(('x', 'y'), fp, 'numpy')
        self.fphi = sp.lambdify(('x', 'y'), fphi, 'numpy')

        self.phixx = sp.lambdify(('x', 'y'), phixx, 'numpy') 
        self.phiyy = sp.lambdify(('x', 'y'), phiyy, 'numpy') 
        self.phixy = sp.lambdify(('x', 'y'), phixy, 'numpy') 

        self.L2phi = sp.lambdify(('x', 'y'), L2phi, 'numpy')

    @cartesian
    def velocity(self, p):
        x, y = p[..., 0], p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.stack((self.u1(x_cpu, y_cpu), self.u2(x_cpu, y_cpu)), axis=-1) 

    @cartesian
    def pressure(self, p):
        x, y = p[..., 0], p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.array(self.p(x_cpu, y_cpu), device=self.device)

    @cartesian
    def grad_pressure(self, p):
        x, y = p[..., 0], p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        val = bm.zeros(p.shape, dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.px(x_cpu, y_cpu), device=self.device)
        val[..., 1] = bm.array(self.py(x_cpu, y_cpu), device=self.device) 
        return val

    @cartesian
    def grad_velocity(self, p):
        x, y = p[..., 0], p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.stack([
            bm.stack((self.u1x(x_cpu, y_cpu), self.u1y(x_cpu, y_cpu)), axis=-1),
            bm.stack((self.u2x(x_cpu, y_cpu), self.u2y(x_cpu, y_cpu)), axis=-1)
        ], axis=-2)  # shape (..., 2, 2)

    @cartesian
    def laplace_velocity(self, p):
        x, y = p[..., 0], p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.stack((self.Lu1(x_cpu, y_cpu), self.Lu2(x_cpu, y_cpu)), axis=-1)


    @cartesian
    def stream_function(self, p):
        x, y = p[..., 0], p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.array(self.phi(x, y), device=self.device)

    @cartesian
    def grad_stream_function(self, p):
        x = p[..., 0]
        y = p[..., 1]
        #sin = bm.sin
        #pi = bm.pi
        #cos = bm.cos
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(p.shape, dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.phix(x, y), device=self.device)
        val[..., 1] = bm.array(self.phiy(x, y), device=self.device) 
        return val

    @cartesian
    def hessian_stream_function(self, p):
        x = p[..., 0]
        y = p[..., 1]
        #sin = bm.sin
        #pi = bm.pi
        #cos = bm.cos
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(p.shape[:-1]+(3, ), dtype=bm.float64, device=self.device)
        val[..., 0] = bm.array(self.phixx(x, y), device=self.device)
        val[..., 1] = bm.array(self.phixy(x, y), device=self.device)
        val[..., 2] = bm.array(self.phiyy(x, y), device=self.device)
        return val



    @cartesian
    def stream_function_source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        #sin = bm.sin
        #pi = bm.pi
        #cos = bm.cos
        return bm.array(self.fphi(x_cpu, y_cpu),device=self.device)


    @cartesian                                                                  
    def source(self, p):                              
        # f = -nu Δu + ∇p; here Δu = 0 since u is harmonic sin(x+y)             
        x, y = p[..., 0], p[..., 1]                                             
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.stack((self.f1(x_cpu, y_cpu), self.f2(x_cpu, y_cpu)), axis=-1)

    @cartesian
    def pressure_source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.array(self.fp(x_cpu, y_cpu), device=self.device)

    @cartesian
    def pressure_neumann_boundary(self, p, n):
        #x = p[..., 0]
        #y = p[..., 1]
        #x_cpu = bm.to_numpy(x)
        #y_cpu = bm.to_numpy(y)
        #bp1  = float(self.viscosity)* bm.einsum('...d,...d->...',self.laplace_velocity(p), n)
        #bp2 = bm.einsum('...d,...d->...', self.source(p), n)
        lap_u = self.laplace_velocity(p)  # (..., d)
        f = self.source(p)                # (..., d)
        return float(self.viscosity) * bm.sum(lap_u * n[:, None, :], axis=-1) + bm.sum(f * n[:, None, :], axis=-1)
        #return bp1+bp2 




class TripleLaplacePDE():
    """
    -\Delta^3 u = f
    """
    def __init__(self, u):
        x = sp.symbols("x")
        y = sp.symbols("y")
        self.su = u

        ux = sp.diff(u, x)
        print('ux', ux)
        print(sp.simplify(ux))
        uy = sp.diff(u, y)
        print('uy', uy)
        print(sp.simplify(uy))
        uxx = sp.diff(ux, x)
        print('uxx', uxx)
        print(sp.simplify(uxx))
        uyy = sp.diff(uy, y)
        print('uyy', uyy)
        print(sp.simplify(uyy))
        uxy = sp.diff(ux, y)
        print('uxy', uxy)
        print(sp.simplify(uxy))

        uxxx = sp.diff(uxx, x)
        print('uxxx', uxxx)
        print(sp.simplify(uxxx))
        uxxy = sp.diff(uxx, y)
        print('uxxy', uxxy)
        print(sp.simplify(uxxy))
        uxyx = sp.diff(uxy, x)
        print('uxyx', uxyx)
        print(sp.simplify(uxyx))
        uxyy = sp.diff(uxy, y)
        print('uxyy', uxyy)
        print(sp.simplify(uxyy))
        uyxx = sp.diff(uxy, x)
        print('uyxx', uyxx)
        print(sp.simplify(uyxx))
        uyxy = sp.diff(uxy, y)
        print('uyxy', uyxy)
        print(sp.simplify(uyxy))
        uyyx = sp.diff(uyy, x)
        print('uyyx', uyyx)
        print(sp.simplify(uyyx))
        uyyy = sp.diff(uyy, y)
        print('uyyy', uyyy)
        print(sp.simplify(uyyy))

        Lu = uxx+uyy
        L2u = sp.diff(Lu, x, 2) + sp.diff(Lu, y, 2)
        L3u = sp.diff(L2u, x, 2) + sp.diff(L2u, y, 2)
        print("-L3u:", -L3u)
        print(sp.simplify(-L3u))

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
    print('uxxxx_sp:', uxxxx_sp)
    print(sp.simplify(uxxxx_sp))
    uyxxx_sp = sp.diff(uyxx_sp, x)
    print('uyxxx_sp:', uyxxx_sp)
    print(sp.simplify(uyxxx_sp))
    uyyxx_sp = sp.diff(uyyx_sp, x)
    print('uyyxx_sp:', uyyxx_sp)
    print(sp.simplify(uyyxx_sp))
    uyyyx_sp = sp.diff(uyyy_sp, x)
    print('uyyyx_sp:', uyyyx_sp)
    print(sp.simplify(uyyyx_sp))
    uyyyy_sp = sp.diff(uyyy_sp, y)
    print('uyyyy_sp:', uyyyy_sp)
    print(sp.simplify(uyyyy_sp))

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
