import numpy as np


from ..mesh.TriangleMesh import TriangleMesh  
from ..mesh.tree_data_structure import Quadtree 

class KelloggData:
    def __init__(self):
        self.a = 161.4476387975881
        self.b = 1

    def init_mesh(self, n=4, meshtype='quadtree'):
        point = np.array([
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)
        if meshtype is 'quadtree':
            cell = np.array([
                (0, 1, 4, 3),
                (1, 2, 5, 4),
                (3, 4, 7, 6),
                (4, 5, 8, 7)], dtype=np.int)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
        elif meshtype is 'tri':
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
        else:
            raise ValueError("".format)
        return mesh

    def diffusion_coefficient(self, p):
        idx = p[:, 0]*p[:, 1] >0
        k = np.ones((p.shape[0], ), dtype=np.float)
        k[idx] = self.a  
        return k

    def subdomain(self, p):
        """
        get the index of the subdomain including point p.
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)
        return np.floor(2*theta/pi)

    def solution(self, p):
    
        x = p[:, 0]
        y = p[:, 1]
    
        pi = np.pi
        cos = np.cos
        sin = np.sin
    
        gamma = 0.1
        sigma = -14.9225565104455152
        rho = pi/4
        r = np.sum(p**2, axis=1) # r=x^2+y^2
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)
        mu = ((theta >= 0) & (theta <= pi/2))*cos((pi/2-sigma)*gamma)*cos((theta-pi/2+rho)*gamma) \
            + ((theta >= pi/2) & (theta <= pi))*cos(rho*gamma)*cos((theta-pi+sigma)*gamma) \
            + ((theta >= pi) & (theta <= 1.5*pi))*cos(sigma*gamma)*cos((theta-pi-rho)*gamma) \
            + ((theta >= 1.5*pi) & (theta <= 2*pi))*cos((pi/2-rho)*gamma)*cos((theta-1.5*pi-sigma)*gamma)
        u = r**gamma*mu
        return u

    def gradient(self, p):
        """The gradient of the exact solution
        """
        x = p[:, 0]
        y = p[:, 1]

        val = np.zeros((len(x), 2), dtype=p.dtype)                      
        
        pi = np.pi
        cos = np.cos
        sin = np.sin
        
        gamma = 0.1
        sigma = -14.9225565104455152
        rho =pi/4        
    
        theta = np.arctan2(p[:, 1], p[:, 0]) #jiaodu
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)    
        t = 1 + (p[:, 1]**2/p[:, 0]**2)
        r = np.sum(p**2, axis = 1)#r = p[:, 0]^2+p[:, 1]^2
        rg = r**gamma 
        ux1 = ((x >= 0.0) & (y >= 0.0))*(rg*gamma/r*cos((pi/2-sigma)*gamma)/r*p[:, 0]*cos((theta-pi/2+rho)*gamma) \
            +rg*cos((pi/2-sigma)*gamma)*sin((theta-pi/2+rho)*gamma)*gamma*p[:, 1]/((p[:, 0]**2)*t))
        uy1 = ((x >= 0.0) & (y >= 0.0))*(rg*gamma/r*cos((pi/2-sigma)*gamma)*cos((theta-pi/2+rho)*gamma)/r*p[:, 1] \
            -rg*cos((pi/2-sigma)*gamma)*sin((theta-pi/2+rho)*gamma)*gamma/(p[:, 0]*t))
        ux2 = ((x <= 0.0) & (y >= 0.0))*(r**(-1.9)*p[:, 0]*gamma*cos(rho*gamma)*cos((theta-pi+sigma)*gamma)\
            +rg*cos(rho*gamma)*sin((theta-pi+sigma)*gamma)*gamma*p[:, 1]/((p[:, 0]**2)*t));
        uy2 = ((x <= 0.0) & (y >= 0.0))*( r**(-1.9)*p[:, 1]*gamma*cos(rho*gamma)*cos((theta-pi+sigma)*gamma)\
            -rg*cos(rho*gamma)*sin((theta-pi+sigma)*gamma)*gamma/(p[:, 0]*t))
        ux3 = ((x <= 0.0) & (y <= 0.0))*(r**(-1.9)*p[:, 0]*gamma*cos(sigma*gamma)*cos((theta-pi-rho)*gamma) \
            +rg*cos(sigma*gamma)*sin((theta-pi-rho)*gamma)*gamma*p[:, 1]/((p[:, 0]**2)*t));
        uy3 = ((x <= 0.0) & (y <= 0.0))*(r**(-1.9)*p[:, 1]*gamma*cos(sigma*gamma)*cos((theta-pi-rho)*gamma) \
            -rg*cos(sigma*gamma)*sin((theta-pi-rho)*gamma)*gamma/(p[:, 0]*t))
        ux4 = ((x >= 0.0) & (y <= 0.0))*(r**(-1.9)*p[:, 0]*gamma*cos((pi/2-rho)*gamma)*cos((theta-3*pi/2-sigma)*gamma) \
            +rg*cos((pi/2-rho)*gamma)*sin((theta-3*pi/2-sigma)*gamma)*gamma*p[:, 1]/((p[:, 0]**2)*t))
        uy4 = ((x >= 0.0) & (y <= 0.0))*(r**(-1.9)*p[:, 1]*gamma*cos((pi/2-rho)*gamma)*cos((theta-3*pi/2-sigma)*gamma)\
            -rg*cos((pi/2-rho)*gamma)*sin((theta-3*pi/2-sigma)*gamma)*gamma/(p[:, 0]*t)) 
        val[:,0] =  ux1+ux2+ux3+ux4
        val[:,1] =  uy1+uy2+uy3+uy4
        return val
    
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p:array object, N*2
        """
        rhs = np.zeros(p.shape[0])
        return rhs

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

class LShapeRSinData:
    def __init__(self):
        pass

    def init_mesh(self, n, meshtype='quadtree'):
        point = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)
        cell = np.array([
            (0, 1, 3, 2),
            (2, 3, 6, 5),
            (3, 4, 7, 6)], dtype=np.int)
        if meshtype is 'quadtree':
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)
    
    def domain(self, n):
        pass

    def solution(self, p):
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        u = (x*x + y*y)**(1/3)*np.sin(2/3*theta)
        return u
    def diffusion_coefficient(self, p):

        return 

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
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        val = np.zeros((len(x),2),dtype=p.dtype)
        val[:, 0] = 2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        val[:, 1] = 2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3)) 
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


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
