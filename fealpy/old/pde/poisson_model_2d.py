import numpy as np
import scipy.io as sio

from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.Quadtree import Quadtree


class ffData:
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='quadtree'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)

    def solution(self, p):
        return np.zeros(p.shape[0:-1])

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        val = np.zeros(p.shape, dtype=p.dtype)
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.ones(x.shape, dtype=np.float)
        flag = np.floor(4*x) + np.floor(4*y)
        isMinus = (flag % 2 == 0)
        val[isMinus] = - 1
        return val

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        val = np.zeros(p.shape[0:-1])
        return val


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
            cell = np.array([
                (1, 4, 0),
                (3, 0, 4),
                (4, 1, 5),
                (2, 5, 1), 
                (4, 7, 3),
                (6, 3, 7),
                (7, 4, 8),
                (5, 8, 4)], dtype=np.int)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
        else:
            raise ValueError("".format)
        return mesh

    def diffusion_coefficient(self, p):
        idx = p[..., 0]*p[..., 1] >0
        k = np.ones(p.shape[:-1], dtype=np.float)
        k[idx] = self.a  
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        is_subdomain = [ p[..., 0]*p[..., 1] >0, p[..., 0]*p[..., 1] < 0]
        return is_subdomain 

    def solution(self, p):
    
        x = p[..., 0]
        y = p[..., 1]
    
        pi = np.pi
        cos = np.cos
        sin = np.sin
    
        gamma = 0.1
        sigma = -14.9225565104455152
        rho = pi/4
        r = np.sqrt(x**2 + y**2) # r=x^2+y^2
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta + 2*pi)

        mu = ((theta >= 0) & (theta < pi/2))*cos((pi/2-sigma)*gamma)*cos((theta-pi/2+rho)*gamma) \
            + ((theta >= pi/2) & (theta < pi))*cos(rho*gamma)*cos((theta-pi+sigma)*gamma) \
            + ((theta >= pi) & (theta < 1.5*pi))*cos(sigma*gamma)*cos((theta-pi-rho)*gamma) \
            + ((theta >= 1.5*pi) & (theta < 2*pi))*cos((pi/2-rho)*gamma)*cos((theta-1.5*pi-sigma)*gamma)

        u = r**gamma*mu
        return u

    def gradient(self, p):
        """The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape, dtype=p.dtype)                      
        
        pi = np.pi
        cos = np.cos
        sin = np.sin
        
        gamma = 0.1
        sigma = -14.9225565104455152
        rho =pi/4        
    
        theta = np.arctan2(y, x) 
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)    
        t = 1 + (y/x)**2
        r = np.sqrt(x**2 + y**2)
        rg = r**gamma 

        ux1= ((x >= 0.0) & (y >= 0.0))*(gamma*rg*cos((pi/2-sigma)*gamma)*(x*cos((theta-pi/2+rho)*gamma)/(r*r) + y*sin((theta-pi/2+rho)*gamma)/(x*x*t)))

        uy1 = ((x >= 0.0) & (y >= 0.0))*(gamma*rg*cos((pi/2-sigma)*gamma)*(y*cos((theta-pi/2+rho)*gamma)/(r*r) - sin((theta-pi/2+rho)*gamma)/(x*t)))

        ux2 = ((x <= 0.0) & (y >= 0.0))*(gamma*rg*cos(rho*gamma)*(x*cos((theta-pi+sigma)*gamma)/(r*r) + y*sin((theta-pi+sigma)*gamma)/(x*x*t)))

        uy2 = ((x <= 0.0) & (y >= 0.0))*(gamma*rg*cos(rho*gamma)*(y*cos((theta-pi+sigma)*gamma)/(r*r) - sin((theta-pi+sigma)*gamma)/(x*t)))

        ux3 = ((x <= 0.0) & (y <= 0.0))*(gamma*rg*cos(sigma*gamma)*(x*cos((theta-pi-rho)*gamma)/(r*r)+y*sin((theta-pi-rho)*gamma)/(x*x*t)))

        uy3 = ((x <= 0.0) & (y <= 0.0))*(gamma*rg*cos(sigma*gamma)*(y*cos((theta-pi-rho)*gamma)/(r*r) - sin((theta-pi-rho)*gamma)/(x*t)))

        ux4 = ((x >= 0.0) & (y <= 0.0))*(gamma*rg*cos((pi/2-rho)*gamma)*(x*cos((theta-3*pi/2-sigma)*gamma)/(r*r)+y*sin((theta-3*pi/2-sigma)*gamma)/(x*x*t)))

        uy4 = ((x >= 0.0) & (y <= 0.0))*(gamma*rg*cos((pi/2-rho)*gamma)*(y*cos((theta-3*pi/2-sigma)*gamma)/(r*r)-sin((theta-3*pi/2-sigma)*gamma)/(x*t)))

        val[...,0] =  ux1+ux2+ux3+ux4
        val[...,1] =  uy1+uy2+uy3+uy4
        return val
    
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p:array object, N*2
        """
        rhs = np.zeros(p.shape[0:-1]) 
        return rhs

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


class LShapeRSinData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='quadtree'):
        point = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)
        if meshtype is 'quadtree':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)
    
    def domain(self, n):
        pass

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        u = (x*x + y*y)**(1/3)*np.sin(2/3*theta)
        return u

    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        rhs = np.zeros(p.shape[0:-1]) 
        return rhs

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        sin = np.sin
        cos = np.cos
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        val = np.zeros(p.shape,dtype=p.dtype)
        val[..., 0] = 2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        val[..., 1] = 2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3)) 
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

class CrackData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='quadtree'):
        if meshtype is 'quadtree':
            r = 1-2**(1/2)/2
            a = 1/2 - 2**(1/2)/2 
            rr = 1/2
            point = np.array([
                (0, -1),
                (-rr, -rr),
                (rr, -rr),
                (-r, -r),
                (0, -r), 
                (r, -r),
                (-1, 0),
                (-r, 0),
                (0, 0),
                (r, 0),
                (1, 0), 
                (r, 0),
                (-r, r),
                (0, r),
                (r, r),
                (-rr,rr),
                (rr,rr),
                (0,1)],dtype=np.float)
            cell = np.array([
                (0, 4, 3, 1),
                (2, 5, 4, 0),
                (1, 3, 7, 6),
                (3, 4, 8, 7),
                (4, 5, 9, 8),
                (5, 2, 10, 9),
                (6, 7, 12, 15),
                (7, 8, 13, 12),
                (8, 11, 14, 13),
                (11, 10, 16, 14),
                (12, 13, 17, 15),    
                (13, 14, 16, 17)], dtype=np.int)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            point = np.array([
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (1, 0),
                (0, 1)], dtype=np.float)

            cell = np.array([
                (2, 1, 0),
                (2, 0, 3),
                (2, 5, 1),
                (2, 4, 5)], dtype=np.int)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)
    def domain(self, n):
        pass 

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]

        r = np.sqrt(x**2 + y**2)
        u = np.sqrt(1/2*(r - x)) - 1/4*r**2
        return u

    def source(self, p):
        rhs = np.ones(p.shape[0:-1])
        return rhs

    def gradient(self, p):
        """the gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]

        r = np.sqrt(x**2 + y**2)
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -0.5*x + (-0.5*x + 0.5*r)**(-0.5)*(0.25*x/r - 0.25)
        val[..., 1] = 0.25*y*(-0.5*x + 0.5*r)**(-0.5)/r - 0.5*y

        return val

    def dirichlet(self, p):
        return self.solution(p)


class CosCosData:
    """
    -\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def init_mesh(self, n=1, meshtype='quadtree'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            print('node',mesh.node.shape)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)   

    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val


    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def neuman(self, p):
        """ Neuman  boundary condition
        """
        pass

    def robin(self, p):
        pass

class SinSinData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='quadtree'):
        point = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)
        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(point, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(point, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)

    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        print('x',x)
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y) - 1
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        uprime[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return uprime

class PolynomialData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        u = (x-x**2)*(y-y**2)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = 2*(y-y**2)+2*(x-x**2)
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[..., 0] = (1-2*x)*(y-y**2)
        uprime[..., 1] = (1-2*y)*(x-x**2)
        return uprime

    def is_boundary(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)


class ExpData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        u = np.exp(x**2+y**2)
        return u

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = -(4*x**2+4*y**2+4)*(np.exp(x**2+y**2))
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float)
        uprime[..., 0] = 2*x*(np.exp(x**2+y**2))
        uprime[..., 1] = 2*y*(np.exp(x**2+y**2))
        return uprime

    def is_boundary(self, p):
        eps = 1e-14 
        return (p[:,0] < eps) | (p[:,1] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] > 1.0 - eps)
        
class sp:
    def __init__(self):
        self.N = 128
        k1 =np.arange(self.N/2+1)
        k2=np.arange(1-self.N/2,0,1)
        self.kk =np.append(k1,k2)

        data = sio.loadmat('u_128.mat')
        self.u1 = data['phi_a']
 
   

    def f(self,x,y):
        N =self.N
        kk =self.kk
        f = 0
        u1 = self.u1
        u1 = np.fft.ifft2(u1)
        for i1 in range(N):
            for i2 in range(N):
                f += u1[i1,i2]*exp(1j*(kk[i1]*x+kk[i2]*y))
        #f =sum(u1@exp(1j*(kk*x.T+kk*y.T)),axis=0)
        return f


    
    def solution(self, p):
        x = p[..., 0]
        x = x.reshape(-1)
        y = p[..., 1]
        y = y.reshape(-1)
        u =self.f(x,y).real
        u = u.reshape(10,-1)
        return u
        
    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        pass
    def source(self, p):
        pass 
    def gradient(self, p):
        pass        
    def is_boundary(self, p):
        pass        
                
             
               
