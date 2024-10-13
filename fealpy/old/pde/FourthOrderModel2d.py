import numpy as np

class PolynomialData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        r = x*x*(x-1.0)*(x-1.0)*y*y*(y-1.0)*(y-1.0) 
        return r

    def gradient(self, p):
        x = p[:, 0]
        y = p[:, 1]
        val = np.zeros((len(x), 2), dtype=p.dtype)
        val[:,0] = 2*(x**2-x)*(2*x-1.0)*y*y*(y-1.0)*(y-1.0)
        val[:,1] = x*x*(x-1.0)*(x-1.0)*2*(y**2-y)*(2*y-1.0)
        return val


    def laplace(self, p):
        x = p[:, 0]
        y = p[:, 1]
        r = (12*x**2 - 12*x + 2)*(y**4 - 2*y**3 + y**2)
        r += (x**4 - 2*x**3 + x**2)*(12*y**2 - 12*y +2)
        return r


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self, p):
        x = p[:, 0]
        y = p[:, 1]
        r = 24.0*(x*x*(x-1.0)*(x-1.0) + y*y*(y-1.0)*(y-1.0))
        r += 2.0*(12.0*x*x - 12.0*x +2.0)*(12.0*y*y - 12.0*y + 2.0)
        return r

class SinSinData:
    def __init__(self):
        pass

    def solution(self, p):
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        r = np.sin(pi*x)*np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*y)
        return r

    def gradient(self, p):
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        val = np.zeros((len(x),2), dtype=p.dtype)
        val[:,0] = 2*pi*np.sin(pi*x)*np.cos(pi*x)*np.sin(pi*y)*np.sin(pi*y)
        val[:,1] = 2*pi*np.sin(pi*x)*np.sin(pi*x)*np.sin(pi*y)*np.cos(pi*y)
        return val


    def laplace(self, p):
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        r = 2*pi**2*np.cos(pi*y)**2*np.sin(pi*x)**2
        r += 2*pi**2*np.cos(pi*x)**2*np.sin(pi*y)**2 
        r -= 4*pi**2*np.sin(pi*x)**2*np.sin(pi*y)**2
        return r

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def source(self, p):
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        pi4 = pi**4
        r1 = np.sin(pi*x)**2
        r2 = np.cos(pi*x)**2
        r3 = np.sin(pi*y)**2
        r4 = np.cos(pi*y)**2
        r = 8*pi4*r2*r4 - 16*pi4*r4*r1 - 16*pi4*r2*r3 + 24*pi4*r1*r3
        return r

