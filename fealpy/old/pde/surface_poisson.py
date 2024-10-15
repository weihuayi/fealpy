import numpy as np

from fealpy.decorator import cartesian

class SphereSimpleData():
    def __init__(self, k=2):
        from fealpy.geometry import SphereSurface
        self.surface = SphereSurface()
        self.k = k

    def domain(self):
        return self.surface

    def init_mesh(self, n=0, meshtype='tri', returnnc=False, p=None):
        mesh = self.surface.init_mesh(meshtype=meshtype, returnnc=returnnc, p=p)
        mesh.uniform_refine(n)
        return mesh

    @cartesian
    def solution(self, p):
        """ The exact solution
        """
        k = self.k
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.power(x, k)+np.power(y, k)+np.power(z, k)
        return u
    def integrate(self):
        k = self.k
        if k==0:
            return 12*np.pi
        if k==2:
            return 4*np.pi
        if k==4:
            return 12*np.pi/5
        if (k%2)==1:
            return 0



class SphereSinSinSinData():
    def __init__(self):
        from fealpy.geometry import SphereSurface
        self.surface = SphereSurface()

    def domain(self):
        return self.surface

    def init_mesh(self, n=0, meshtype='tri', returnnc=False, p=None):
        mesh = self.surface.init_mesh(meshtype=meshtype, returnnc=returnnc, p=p)
        mesh.uniform_refine(n)
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
    def gradient(self, p):
        """ The Gradu of the exact solution
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
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
        
        grad = np.zeros(p.shape, dtype=np.float64)
        grad[..., 0] = valx
        grad[..., 1] = valy
        grad[..., 2] = valz
        return grad

    def normal(self,p):
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        n = np.zeros(p.shape, dtype=np.float)
        n[..., 0] = x
        n[..., 1] = y
        n[..., 2] = z
        val = np.sqrt(np.sum(n**2, axis=-1))
        n[...,0] /= val
        n[...,1] /= val
        n[...,2] /= val
        return n

class EllipsoidSinSinSinData():
    def __init__(self):
        from fealpy.geometry import SphereSurface,EllipsoidSurface
        self.surface = EllipsoidSurface()
        """
        x^2/81 + y^2/9 +z^2 =1
        """

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
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin

        t1 = sin(pi*x)*sin(pi*y)*sin(pi*z)
        t2 = cos(pi*x)*cos(pi*y)*sin(pi*z)
        t3 = cos(pi*x)*sin(pi*y)*cos(pi*z)
        t4 = sin(pi*x)*cos(pi*y)*cos(pi*z)
        
        u1 = cos(pi*x)*sin(pi*y)*sin(pi*z)
        u2 = sin(pi*x)*cos(pi*y)*sin(pi*z)
        u3 = sin(pi*x)*sin(pi*y)*cos(pi*z)

        r = np.sqrt((x/81)**2 + (y/9)**2 + z**2)
        divn = (1/81+1/9+1)/r-(x**2/(81**3)+y**2/(81*9)+z**2)/(r**3)

        a1 = 2*pi**2*t1
        a2 = divn*pi*(x*u1/81+y*u2/9+z*u3)/r
        a3 = 2*pi**2*(x*y*t2/(81*9)+x*z*t3/81+y*z*t4/9)/(r**2)
        rhs = a1 + a2 + a3
        return rhs


    @cartesian
    def gradient(self, p):
        """ The Gradu of the exact solution
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        
        t1 = cos(pi*x)*sin(pi*y)*sin(pi*z)
        t2 = sin(pi*x)*cos(pi*y)*sin(pi*z)
        t3 = sin(pi*x)*sin(pi*y)*cos(pi*z)
        r = np.sqrt((x/81)**2 + (y/9)**2 + z**2)


        valx = pi*(t1 - (t1*(x/81)**2 + t2*x*y/(81*9) + t3*x*z/81)/r**2)
        valy = pi*(t2 - (t1*x*y/(81*9) + t2*y**2/81 + t3*y*z/9)/r**2)
        valz = pi*(t3 - (t1*x*z/81 + t2*y*z/9 +t3*z**2)/(r**2))
        
        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., 0] = valx
        grad[..., 1] = valy
        grad[..., 2] = valz
        return grad
