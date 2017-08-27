import numpy as np

class LShapeRSinData:
    def __init__(self):
        pass

    def solution(self, p):
        x = p[:, 0]
        y = p[:, 1]
        theta = np.arctan2(y, x)
        u = (x*x + y*y)**(1/3)*np.sin(2/3*theta)
        return u

    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        rhs = np.zeros(p.shape[0]) 
        return rhs

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        sin = np.sin
        cos = np.cos
        x = p[:, 0]
        y = p[:, 1]
        theta = np.arctan2(y, x)
        val = np.zeros((len(x),2),dtype=p.dtype)
        val[:, 0] =2/3*x*(x**2 + y**2)**(-2/3)*sin(2/3*theta)
        -2/3*y*(x**2 +y**2)**(1/3)*np.cos(2/3*theta)/(x**2*(1 + y**2/x**2))
        val[:, 1] = 2/3*y*(x**2 + y**2)**(-2/3)*np.sin(2/3*theta) + (2/3)*(x**2 +
        y**2)**(1/3)*np.cos(2/3*theta)/(x*(1 + y**2/x**2))
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

    def is_boundary(self, p):
        eps = 1e-14
        return (p[:, 0] < -1.0 + eps)| (p[:, 1] > 1.0 - eps) | ((p[:, 1]< 1.0 - eps) & (p[:,0]>-eps)) | ((p[:, 1] < -eps) & (p[:, 0] > 1.0 -eps))

class CosCosData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        u = np.cos(pi*x)*np.cos(pi*y)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        rhs = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[:, 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        uprime[:, 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return uprime

    def is_boundary(self, p):
        eps = 1e-14 
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)

class PolynomialData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        u = (x-x**2)*(y-y**2)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[:, 0]
        y = p[:, 1]
        rhs = 2*(y-y**2)+2*(x-x**2)
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[:, 0] = (1-2*x)*(y-y**2)
        uprime[:, 1] = (1-2*y)*(x-x**2)
        return uprime

    def is_boundary(self, p):
        eps = 1e-14 
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)


class ExpData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        u = np.exp(x**2+y**2)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[:, 0]
        y = p[:, 1]
        rhs = -(4*x**2+4*y**2+4)*(np.exp(x**2+y**2))
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[:, 0] = 2*x*(np.exp(x**2+y**2))
        uprime[:, 1] = 2*y*(np.exp(x**2+y**2))
        return uprime

    def is_boundary(self, p):
        eps = 1e-14 
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)
