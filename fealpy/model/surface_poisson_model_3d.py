import numpy as np
from ..mesh.TriangleMesh import TriangleMesh 

class SphereData(object):
    def __init__(self, a = 0.6):
        self.a = a

    def solution(self, p):
        """ The exact solution 
        """
        a = self.a
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        
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
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

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
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        pi = np.pi

        grad = np.zeros(p.shape, dtype=np.float)
        
        return grad
    
    def dirichlet(self,p):
        pass


class SquaredSphereData(object):
    def __init__(self):
        pass
    
    def solution(self,p):
        """ The exact solution
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*z)
        return u


    def source(self, p):
        """ The right hand side of Possion equation
            INPUT:
            p: array object, N*3
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
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
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
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
        
        gradus = np.zeros(p.shape, dtype=np.float)
        gradus[:, 0] = valx/(x**14+y**14+z**14)
        gradus[:, 1] = valy/(x**14+y**14+z**14)
        gradus[:, 2] = valz/(x**14+y**14+z**14)
        return gradus


class ElipsoidData(object):
    def __init__(self):
        self.a = 9
        self.b = 3
        self.c = 1

    def solution(self,p):
        """ The exact solution 
        """
        a = self.a
        b = self.b
        c = self.c

        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
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
        cos = np.cos
        sin = np.sin

        t1 = (b**2*z**2+y**2*c**2)*a**4 + (y**2*c**4+z**2*b**4)*a**2 + b**2*c**2*x**2*(b**2+c**2)
        t2 = a**2*sin(x)*(a**4*(y**2*c**4+z**2*b**4)+x**2*b**4*c**4*(y**2*c**4+z**2*b**4))
        t3 = (a**4*(y**2*c**4+z**2*b**4)+x**2*b**4*c**4)**2

        rhs = a**2*((c**4*t1*b**4*x*cos(x) + t2))/t3
        return rhs

    
    def gradient(self,p):
        """The Gradu of the exact solution on surface
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        cos = np.cos
        sin = np.sin
        
        val1 = x**2/a**4 +y**2/b**4 + z**2/c**4
        valx = cos(x) - (cos(x)*x**2/val1*a**4)
        valy = -(cos(x)*x*y)/(val1*a**2*b**2)
        valz = -(cos(x)*x*z)/(val1*a**2*c**2)

        gradus = np.zeros(p.shape, dtype=np.float)
        gradus[:, 0] = valx
        gradus[:, 1] = valy
        gradus[:, 2] = valz        
        return gradus


class OrthocircleData(object):
    def __init__(self):
        pass

    def solution(self,p):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        u = np.exp(x+y+z)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        
        s1 = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        s2 = x**6-2*x**4+(-y**3+y-z**3+z)*x**3+x**2+(y**3-y+z**3-z)*x+y**6-2*y**4+(-z**3+z)*y**3+y**2+(z**3-z)*y+z**2+z**6-2*z**4
        s3 = (z**2-2/3+y**2)*x**6+(4/3-2*y**2-2*z**2)*x**4+(2*z**2+z**6-2*z**4-2/3+2*y**2+y**6-2*y**4)*x**2 \ + (z**2-2/3)*y**6+(4/3-2*z**2)*y**4+(2*z**2+z**6-2*z**4-2/3)*y**2-2/3*z**2*(z-1)**2*(z+1)**2 
        rhs = -1/(s1)**2*2*(s2*s1-3/2*s3*(x**3-x+y**3-y+z**3-z)*np.exp(x+y+z))
        return rhs

    def gradus(self,p):
        """The Gradu of the exact solution on surface
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        val = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        t1 = (y**3-y+z**3-z)*x**3+(-y**3+y-z**3+z)*x-y**6+2*y**4-y**2-z**6+2*z**4-z**2
        t2 = (-x**3+x-z**3+z)*y**3+(x**3-x+z**3-z)*y+x**6-2*x**4+x**2+z**6-2*z**4+z**2
        t3 = (-x**3+x-y**3+y)*z**3+(x**3-x+y**3-y)*z-2*x**4+x**2+y**6+y**2-2*y**4
        gradus = np.zeros(p.shape, dtype=np.float)

        gradus[:, 0] = -t1*np.exp(x+y+z)/val
        gradus[:, 1] = t2*np.exp(x+y+z)/val
        gradus[:, 2] = t3*np.exp(x+y+z)/val
        return gradus


class QuarticsData(object):
    def __init__(self):
        pass
    
    def solution(self,p):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        u = np.exp(x+y+z)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*3 
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        
        s1 = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        s2 = x**6-2*x**4+(-y**3+y-z**3+z)*x**3+x**2+(y**3-y+z**3-z)*x+y**6-2*y**4+(-z**3+z)*y**3+y**2+(z**3-z)*y+z**2+z**6-2*z**4
        s3 = (z**2-2/3+y**2)*x**6+(4/3-2*y**2-2*z**2)*x**4+(2*z**2+z**6-2*z**4-2/3+2*y**2+y**6-2*y**4)*x**2 \
            + (z**2-2/3)*y**6+(4/3-2*z**2)*y**4+(2*z**2+z**6-2*z**4-2/3)*y**2-2/3*z**2*(z-1)**2*(z+1)**2
        rhsF = -1/(s1)**2*2*(s2*s1-3/2*s3*(x**3-x+y**3-y+z**3-z)*np.exp(x+y+z))
        return rhsF    


    def gradu(self,p):
        """The Gradu of the exact solution
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        gradu = np.zeros(p.shape, dtype=np.float) 
        gardu[:, 0] = np.exp(x+y+z)
        gradu[:, 1] = np.exp(x+y+z)
        gradu[:, 2] = np.exp(x+y+z)
        return gradu 

    def gradus(self,p):
        """The Gradu of the exact solution on surface
        """
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        val = x**6-2*x**4+x**2+y**6-2*y**4+y**2+z**6-2*z**4+z**2
        t1 = (y**3-y+z**3-z)*x**3+(-y**3+y-z**3+z)*x-y**6+2*y**4-y**2-z**6+2*z**4-z**2
        t2 = (-x**3+x-z**3+z)*y**3+(x**3-x+z**3-z)*y+x**6-2*x**4+x**2+z**6-2*z**4+z**2
        t3 = (-x**3+x-y**3+y)*z**3+(x**3-x+y**3-y)*z-2*x**4+x**2+y**6+y**2-2*y**4
        gradus = np.zeros(p.shape, dtype=np.float)

        gradus[:, 0] = -t1*np.exp(x+y+z)/val
        gradus[:, 1] = t2*np.exp(x+y+z)/val
        gradus[:, 2] = t3*np.exp(x+y+z)/val
        return gradus

    def dirichlet(self,p):
        pass


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

    def gradu(self,p):
        """The Gradu of the exact solution
        """
        pass
    
    def gradus(self,p):
        """The Gradu of the exact solution on surface
        """
        pass

    def dirichlet(self,p):
        pass

class TorusData(object):
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


    def gradu(self,p):
        """The Gradu of the exact solution
        """
        pass

    def gradus(self,p):
        """The Gradu of the exact solution on surface
        """
        pass

    def dirichlet(self,p):
        pass

class HearsurfacetData(object):
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


    def gradu(self,p):
        """The Gradu of the exact solution
        """
        pass
    def gradus(self,p):
        """The Gradu of the exact solution on surface
        """
        pass

    def dirichlet(self,p):
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


    def gradu(self,p):
        """The Gradu of the exact solution
        """
        pass
    def gradus(self,p):
        """The Gradu of the exact solution on surface
        """
        pass

    def dirichlet(self,p):
        pass
























