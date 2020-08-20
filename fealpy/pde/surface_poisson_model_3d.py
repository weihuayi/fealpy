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
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3
        """
        p, _ = self.surface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        
        t1 = sin(pi*x)*sin(pi*y)*sin(pi*z)*pi
        t2 = sin(pi*x)*cos(pi*z)*cos(pi*y)*pi*y*z + sin(pi*y)*cos(pi*z)*cos(pi*x)*pi*x*z + cos(pi*x)*sin(pi*z)*cos(pi*y)*pi*x*y
        t3 = sin(pi*x)*sin(pi*y)*cos(pi*z)*z + sin(pi*x)*sin(pi*z)*cos(pi*y)*y + sin(pi*y)*cos(pi*x)*sin(pi*z)*x
        r = x**2 + y**2 + z**2
        rhs = 2*pi*(t1 + (t2 + t3)/r) 
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

        valx = pi*(t1 - (t1*x**2 + t2*x*y + t3*x*z))
        valy = pi*(t2 - (t1*x*y + t2*y**2 + t3*y*z))
        valz = pi*(t3 - (t1*x*z + t2*y*z +t3*z**2))
        
        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., 0] = valx/r
        grad[..., 1] = valy/r
        grad[..., 2] = valz/r
        return grad  


class HeartSurfacetData(object):
    def __init__(self):
        from fealpy.geometry import HeartSurface
        self.surface = HeartSurface() 

    def init_mesh(self):
        return self.surface.init_mesh()

    def domain(self):
        return self.surface

    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = x*y
        return u
    
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        
        val1 = 7*z**8-25*x*z**6+31*x**2*z**4+3*y**2*z**4+8*z**6-15*x**3*z**2-7*x*y**2*z**2-21*x*z**4
        val2 = 2*x**4+2*x**2*y**2+16*x**2*z**2+2*y**2*z**2+2*z**4-3*x**3-3*x*y**2-3*x*z**2
        val = 4*z**6-8*x*z**4+4*x**2*z**2+5*z**4-6*x*z**2+x**2+y**2+z**2
        rhs = -2*y*(val1+val2)/val**2
        return rhs


    def gradient(self, p):
        """The Gradu of the exact solution on surface
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 4*z**6-8*x*z**4+4*x**2*z**2+5*z**4-6*x*z**2+x**2+y**2+z**2
        valx = y*(4*z**6-8*x*z**4+4*x**2*z**2+4*z**4-3*x*z**2-x**2+y**2+z**2)
        valy = 4*x*z**6-8*x**2*z**4+4*x**3*z**2+5*x*z**4-6*x**2*z**2+y**2*z**2+x**3-x*y**2+x*z**2
        valz = y*(-z**2+2*x)*z*(-2*z**2+2*x-1)
        
        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., 0] = valx/val
        grad[..., 1] = valy/val
        grad[..., 2] = valz/val
        return grad


class ElipsoidSurfaceData(object):
    def __init__(self):
        from ..mesh.level_set_function import EllipsoidSurface
        self.surface = EllipsoidSurface()
        self.a = 9
        self.b = 3
        self.c = 1

    def solution(self,p):
        """ The exact solution 
        """
        a = self.a
        b = self.b
        c = self.c

        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = np.sin(x)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        a = self.a
        b = self.b
        c = self.c
        
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        cos = np.cos
        sin = np.sin
        
        t = a**4*b**4*z**2 + a**4*c**4*y**2 + b**4*c**4*x**2
        t1 = a**6*b**8*sin(x)*z**4 + 2*a**6*sin(x)*y**2*z**2*b**4*c**4+a**6*c**8*sin(x)*y**4+b**8*sin(x)*x**2*z**2*a**2*c**4
        t2 = c**8*sin(x)*x**2*y**2*a**2*b**4+cos(x)*x*z**2*a**4*b**6*c**4+cos(x)*x*y**2*a**4*b**4*c**6+b**8*cos(x)*x*z**2*a**2*c**4
        t3 = c**8*cos(x)*x*y**2*a**2*b**4+b**8*cos(x)*x**3*c**6+c**8*cos(x)*x**3*b**6

        rhs = (a**2*(t1+t2+t3))/t**2
        return rhs

    
    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        a = self.a
        b = self.b
        c = self.c

        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        cos = np.cos
        sin = np.sin
       
        val = a**4*b**4*z**2 + a**4*c**4*y**2 + b**4*c**4*x**2
        valx = cos(x)*a**4*(b**4*z**2+c**4*y**2)
        valy = cos(x)*x*y*a**2*b**2*c**4
        valz = cos(x)*x*z*a**2*b**4*c**2

        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., 0] = valx/val
        grad[..., 1] = -valy/val
        grad[..., 2] = -valz/val       
        return grad


class ToruSurfacesData(object):
    def __init__(self):
        pass

    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]        
        u = x*y
        return u
    
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]

        t = np.sqrt(x**2+y**2)
        t1 = (-x**2-y**2-z**2+8*t-16)**4*t**2

        t2 = -32768+9*t*x**2*z**4+9*t*y**2*z**4+1248*t*x**2*z**2+1248*t*y**2*z**2+61440*t-384*z**4-8*z**6
        t3 = 9*t*x**4*y**2+9*t*x**2*y**4+336*t*z**4+8448*t*z**2+3*t*z**6-96*x**2*z**4-96*y**2*z**4+3*t*x**6+3*t*y**6
        t4 = 9*t*x**4*z**2+912*t*x**4+912*t*y**4+9*t*y**4*z**2+1824*t*x**2*y**2+18*t*x**2*z**2*y**2-168*y**4*z**2-80*y**6-240*x**4*y**2-240*x**2*y**4-11520*x**2*y**2
        t5 = -168*x**4*z**2-336*x**2*y**2*z**2-5760*y**4+21760*t*x**2+21760*t*y**2-4608*x**2*z**2-4608*y**2*z**2-5760*x**4-80*x**6-6144*z**2-49152*x**2-49152*y**2
        rhs = 2*(t2+t3+t4+t5)*x*(-4+t)*y/t1
        return rhs

    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        t = np.sqrt(x**2+y**2)

        val1 = (-x**2-y**2-z**2+8*t-16)*t**2
        val2 = t*(-x**2-y**2-z**2+8*t-16)
        valx = -x**4+x**2*z**2+y**4+y**2*z**2+8*t*x**2-8*t*y**2-16*x**2+16*y**2
        valy = -x**4-x**2*z**2+y**4-y**2*z**2+8*t*x**2-8*t*y**2-16*x**2+16*y**2
        valz = 2*z*x*(-4+t)*y

        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., :, 0] = -valx*y/val1
        grad[..., :, 1] = valy*x/val1
        grad[..., :, 2] = valz/val2
        return grad


class SquaredSphereData(object):
    def __init__(self):
        pass
    
    def solution(self,p):
        """ The exact solution
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*z)
        return u


    def source(self, p):
        """ The right hand side of Possion equation
            INPUT:
            p: array object, N*3
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        pi = np.pi
        cos = np.cos
        sin = np.sin

        t1 = pi*(x**14+y**14+z**14)**2*sin(pi*z)+7/2*((z**6+y**6)*x**14 + (y**14+z**14)*x**6 + y**6*z**6*(z**8+y**8))*z**7*cos(pi*z)
        t2 = (z**6+y**6)*x**14 + (y**14+z**14)*x**6 + y**6*z**6*(z**8+y**8)
        t3 = 2/7*pi*cos(pi*z)*z**7*(x**14+y**14+z**14)

        t4 = (7/2*z**6 + 7/2*y**6)*x**14 + (7/2*z**14 + 7/2*y**14)*x**6 + 7/2*y**6*z**6*(z**8+y**8)
        t5 = pi*cos(pi*z)*z**7*(x**14+y**14+z**14)
        t6 = sin(pi*z)*y**7*pi*cos(pi*y)*(x**14+y**14+z**14)
        
        rhs = (1/(x**14+y**14+z**14)) *2*pi*((t1*sin(pi*y) + 7/2*(t2*sin(pi*z) + t3)*y**7*cos(pi*y))*sin(pi*x) + ((t4*sin(pi*z) + t5)*sin(pi*y) + t6)*x**7*cos(pi*x))
        
        return rhs

    def gradient(self,p):
        """ The exace solution on surface
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        pi = np. pi
        cos = np.cos
        sin = np.sin


        valx1 = ((-y**14-z**14)*cos(pi*x)*sin(pi*z)+sin(pi*x)*cos(pi*z)*z**7*x**7)*sin(pi*y)
        valx2 = y**7*sin(pi*x)*cos(pi*y)*sin(pi*z)*x**7
        valx = -(valx1+valx2)*pi

        valy1 = ((x**14+z**14)*cos(pi*x)*sin(pi*z)-sin(pi*y)*cos(pi*z)*z**7*y**7)*sin(pi*x)
        valy2 = x**7*cos(pi*x)*sin(pi*y)*sin(pi*z)*y**7
        valy = (valy1-valy2)*pi

        valz1 = ((x**14+y**14)*cos(pi*z)*sin(pi*y)-cos(pi*y)*sin(pi*z)*y**7*z**7)*sin(pi*x)
        valz2 = x**7*cos(pi*x)*sin(pi*y)*sin(pi*z)*z**7
        valz = (valz1 - valz2)*pi
        
        grad = np.zeros(p.shape, dtype=np.float)
        grad[..., :, 0] = valx/(x**14+y**14+z**14)
        grad[..., :, 1] = valy/(x**14+y**14+z**14)
        grad[..., :, 2] = valz/(x**14+y**14+z**14)
        return grad


class OrthocircleData(object):
    def __init__(self):
        pass

    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]

        u = np.exp(x+y+z)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        
        s1 = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        s2 = x**6-2*x**4+(-y**3+y-z**3+z)*x**3+x**2+(y**3-y+z**3-z)*x+y**6-2*y**4+(-z**3+z)*y**3+y**2+(z**3-z)*y+z**2+z**6-2*z**4
        s3 = (z**2-2/3+y**2)*x**6+(4/3-2*y**2-2*z**2)*x**4+(2*z**2+z**6-2*z**4-2/3+2*y**2+y**6-2*y**4)*x**2 \
            + (z**2-2/3)*y**6+(4/3-2*z**2)*y**4+(2*z**2+z**6-2*z**4-2/3)*y**2-2/3*z**2*(z-1)**2*(z+1)**2 
        rhs = -1/(s1)**2*2*(s2*s1-3/2*s3*(x**3-x+y**3-y+z**3-z)*np.exp(x+y+z))
        return rhs

    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]

        val = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        t1 = (y**3-y+z**3-z)*x**3+(-y**3+y-z**3+z)*x-y**6+2*y**4-y**2-z**6+2*z**4-z**2
        t2 = (-x**3+x-z**3+z)*y**3+(x**3-x+z**3-z)*y+x**6-2*x**4+x**2+z**6-2*z**4+z**2
        t3 = (-x**3+x-y**3+y)*z**3+(x**3-x+y**3-y)*z-2*x**4+x**2+y**6+y**2-2*y**4
        grad = np.zeros(p.shape, dtype=np.float)

        grad[..., :, 0] = -t1*np.exp(x+y+z)/val
        grad[..., :, 1] = t2*np.exp(x+y+z)/val
        grad[..., :, 2] = t3*np.exp(x+y+z)/val
        return grad


class QuarticsData(object):
    def __init__(self):
        pass
    
    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]

        u = np.exp(x+y+z)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        
        s1 = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        s2 = x**6-2*x**4+(-y**3+y-z**3+z)*x**3+x**2+(y**3-y+z**3-z)*x+y**6-2*y**4+(-z**3+z)*y**3+y**2+(z**3-z)*y+z**2+z**6-2*z**4
        s3 = (z**2-2/3+y**2)*x**6+(4/3-2*y**2-2*z**2)*x**4+(2*z**2+z**6-2*z**4-2/3+2*y**2+y**6-2*y**4)*x**2 \
            + (z**2-2/3)*y**6+(4/3-2*z**2)*y**4+(2*z**2+z**6-2*z**4-2/3)*y**2-2/3*z**2*(z-1)**2*(z+1)**2
        rhs = -1/(s1)**2*2*(s2*s1-3/2*s3*(x**3-x+y**3-y+z**3-z)*np.exp(x+y+z))
        return rhs    


    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]

        val = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        t1 = (y**3-y+z**3-z)*x**3+(-y**3+y-z**3+z)*x-y**6+2*y**4-y**2-z**6+2*z**4-z**2
        t2 = (-x**3+x-z**3+z)*y**3+(x**3-x+z**3-z)*y+x**6-2*x**4+x**2+z**6-2*z**4+z**2
        t3 = (-x**3+x-y**3+y)*z**3+(x**3-x+y**3-y)*z-2*x**4+x**2+y**6+y**2-2*y**4
        grad = np.zeros(p.shape, dtype=np.float)

        grad[..., :, 0] = -t1*np.exp(x+y+z)/val
        grad[..., :, 1] = t2*np.exp(x+y+z)/val
        grad[..., :, 2] = t3*np.exp(x+y+z)/val
        return grad


class DoubleTorusData(object):
    def __init__(self):
        pass

    def solution(self,p):
        """ The exact solution 
        """
        pass
    
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        pass
    
    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        pass


class Mcmullenk3Data(object):
    def __init__(self):
        pass

    
    def solution(self,p):
        """ The exact solution 
        """
        pass

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        pass

    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        pass

class SphereData(object):
    def __init__(self, a = 0.6):
        self.a = a

    def solution(self, p):
        """ The exact solution 
        """
        a = self.a
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        
        phi = np.arctan2(y,x)
        phi = (phi >= 0)*phi + (phi < 0)*(phi + 2*pi)
        
        theta = np.arccos(z)
        u = np.sin(theta)**a*np.sin(a*phi)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        a = self.a
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]

        pi = np.pi
        phi = np.arctan2(y,x)
        phi = (phi >= 0)*phi + (phi < 0)*(phi + 2*pi)
        
        theta = np.arccos(z)
        rhs = a*(a+1)*(np.sin(theta))**a*np.sin(a*phi)
        return rhs

    def gradient(self, p):
        """ The Gradu of the exact solution
        """
        a = self.a
        x = p[..., :, 0]
        y = p[..., :, 1]
        z = p[..., :, 2]
        pi = np.pi

        grad = np.zeros(p.shape, dtype=np.float)
        
        return grad
    






















