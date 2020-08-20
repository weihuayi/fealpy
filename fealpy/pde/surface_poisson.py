import numpy as np

from fealpy.decorator import cartesian

class SphereSinSinSinData():
    def __init__(self):
        from fealpy.geometry import SphereSurface
        self.surface = SphereSurface()

    def domain(self):
        return self.surface

    def init_mesh(self, n=0):
        mesh = self.surface.init_mesh()
        mesh.uniform_refine(n, self.surface)
        return mesh

    @cartesian
    def solution(self,p):
        """ The exact solution
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*z)
        return u

    @cartesian
    def source(self, p, n=None):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3
        """
        p0, d = self.surface.project(p)


        x = p0[..., 0]
        y = p0[..., 1]
        z = p0[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        
        t1 = sin(pi*x)*sin(pi*y)*sin(pi*z)*pi
        t2 = sin(pi*x)*cos(pi*z)*cos(pi*y)*pi*y*z + sin(pi*y)*cos(pi*z)*cos(pi*x)*pi*x*z + cos(pi*x)*sin(pi*z)*cos(pi*y)*pi*x*y
        t3 = sin(pi*x)*sin(pi*y)*cos(pi*z)*z + sin(pi*x)*sin(pi*z)*cos(pi*y)*y + sin(pi*y)*cos(pi*x)*sin(pi*z)*x
        r = x**2 + y**2 + z**2
        rhs = 2*pi*(t1 + (t2 + t3)/r) 

        if n is not None:
            n0 = self.surface.unit_normal(p)
            H0 = self.surface.hessian(p0)
            e, _ = np.linalg.eig(H0)
            e = e.real
            e /= 1 + d[..., None]*e
            e *= -d[..., None]
            e += 1
            rhs *= np.sum(n*n0, axis=-1)
            rhs *= np.product(e, axis=-1)

        return rhs


    @cartesian
    def gradient(self, p, n=None):
        """ 

        Notes
        -----

        单位球面上的真解
        """
        p0, d = self.surface.project(p)
        x = p0[..., 0]
        y = p0[..., 1]
        z = p0[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        r = x**2 + y**2 + z**2
        
        t1 = cos(pi*x)*sin(pi*y)*sin(pi*z)
        t2 = sin(pi*x)*cos(pi*y)*sin(pi*z)
        t3 = sin(pi*x)*sin(pi*y)*cos(pi*z)

        valx = pi*(t1 - (t1*x**2 + t2*x*y + t3*x*z)/r)
        valy = pi*(t2 - (t1*x*y + t2*y**2 + t3*y*z)/r)
        valz = pi*(t3 - (t1*x*z + t2*y*z +t3*z**2)/r)
        
        val = np.zeros(p0.shape, dtype=np.float64)
        val[..., 0] = valx
        val[..., 1] = valy
        val[..., 2] = valz

        if n is not None:
            H = self.surface.hessian(p)
            n0 = self.surface.unit_normal(p0)
            val -= np.sum(n0*val, axis=-1, keepdims=True)*n0
            val -= np.einsum('..., ...mn, ...n->...m', d, H, val)
            val -= np.sum(n*val, axis=-1, keepdims=True)*n
        return val  
