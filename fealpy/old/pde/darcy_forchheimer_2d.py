import numpy as np

from ..mesh.litest_quadmesh import StructureQuadMesh1
from ..mesh.StructureQuadMesh import StructureQuadMesh
from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.Mesh2d import Mesh2d


class LShapeRSinData():
    def __init__(self, mu, rho, beta, alpha, tol, maxN):
        self.mu = mu
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.tol = tol
        self.maxN = maxN

    def init_mesh(self, n=4):
        node = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float)
        cell = np.array([
            (1, 3, 0),
            (2, 0, 3),
            (3, 6, 2),
            (5, 2, 6),
            (4, 7, 3),
            (6, 3, 7)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh
    
    def f(self, p):
        x = p[..., 0]
        y = p[..., 1]
        mu = self.mu
        rho = self.rho
        beta = self.beta
        pi = np.pi
        sin = np.sin
        cos = np.cos
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        val = np.zeros(p.shape, dtype=p.dtype)
        t1 = -2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        t2 = -2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3))
        val[..., 0] = mu/rho*t1 + beta/rho*np.sqrt(t1**2 + t2**2)*t1 - t1
        val[..., 1] = mu/rho*t2 + beta/rho*np.sqrt(t1**2 + t2**2)*t2 - t2
        return val
    
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        p = (x*x + y*y)**(1/3)*np.sin(2/3*theta)
        return p

    def velocity(self, p):
        sin = np.sin
        cos = np.cos
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        u = np.zeros(p.shape, dtype=p.dtype)
        u[..., 0] = -2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        u[..., 1] = -2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3))
        return u
    
    def g(self, p):
        '''The right hand side of DarcyForchheimer equation
        '''
        sin = np.sin
        cos = np.cos
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        val = 4*(x**2/r + y**2/r - 1)*sin(2*theta/3)/(3*r**(2/3))
        return val
    
    def grad_pressure(self, p):
        sin = np.sin
        cos = np.cos
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        u = np.zeros(p.shape, dtype=p.dtype)
        u[..., 0] = 2*(x*sin(2*theta/3) - y*cos(2*theta/3))/(3*r**(2/3))
        u[..., 1] = 2*(x*cos(2*theta/3) + y*sin(2*theta/3))/(3*r**(2/3))
        return u

    def Neumann_boundary(self, p):
        sin = np.sin
        cos = np.cos
        pi = np.pi
        z = np.zeros(p.shape[0],)
        x = p[..., 0]
        y = p[..., 1]
        
        idx = np.nonzero(abs(x - 1) < np.spacing(0))
        z[idx] = -2*(sin(2*np.arctan2(y[idx], 1)/3) - y[idx]*cos(2*np.arctan2(y[idx], 1)/3))/(3*(1 + y[idx]**2)**(2/3))

        idx = np.nonzero(abs(x - 0) < np.spacing(0))
        z[idx] = -2*(sin(2*np.arctan2(y[idx], 0)/3) - y[idx]*cos(2*np.arctan2(y[idx], 0)/3))/(3*(y[idx]**2)**(2/3))

        idx = np.nonzero(abs(x + 1) < np.spacing(0))
        z[idx] = 2*(sin(2*np.arctan2(y[idx], -1)/3) - y[idx]*cos(2*np.arctan2(y[idx], -1)/3))/(3*(1 + y[idx]**2)**(2/3))

        idx = np.nonzero(abs(y - 1) < np.spacing(0))
        z[idx] = -2*(x[idx]*cos(2*np.arctan2(1, x[idx])/3) + sin(2*np.arctan2(1, x[idx])/3))/(3*(x[idx]**2 + 1)**(2/3))

        idx = np.nonzero(abs(y - 0) < np.spacing(0))
        z[idx] = 2*(x[idx]*cos(2*np.arctan2(0, x[idx])/3) + sin(2*np.arctan2(0, x[idx])/3))/(3*(x[idx]**2)**(2/3))

        idx = np.nonzero(abs(y + 1) < np.spacing(0))
        z[idx] = 2*(x[idx]*cos(2*np.arctan2(-1, x[idx])/3) + sin(2*np.arctan2(-1, x[idx])/3))/(3*(x[idx]**2 + 1)**(2/3))

        return z














class PolyData:
    """
    u = (np.sin(pi*x)*np.cos(pi*y),np.cos(pi*x)*np.sin(pi*y))^t
    p = x*(1-x)*y*(1-y)
    g = 2*pi*np.cos(pi*x)*np.cos(pi*y)
    f = ((mu/k + rho*beta*np.sqrt(np.sin(pi*x)**2*np.cos(pi*y)**2 + np.cos(pi*x)**2*np.sin(pi*y)**2))*np.sin(pi*x)*np.cos(pi*y) + (1-2*x)*y*(1-y),
         (mu/k + rho*beta*np.sqrt(np.sin(pi*x)**2*np.cos(pi*y)**2 + np.cos(pi*x)**2*np.sin(pi*y)**2))*np.cos(pi*x)*np.sin(pi*y) + x(1-x)(1-2y))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = 2*pi*np.cos(pi*x)*np.cos(pi*y)
        return rhs


    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:,0] = np.sin(pi*x)*np.cos(pi*y)
        val[:,1] = np.cos(pi*x)*np.sin(pi*y)
        return val

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.cos(pi*y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.sin(pi*y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*y*(1-y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(pi*x)
        t1 = np.cos(pi*x)
        t2 = np.sin(pi*y)
        t3 = np.cos(pi*y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = m*t0*t3 + (1-2*x)*y*(1-y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(pi*x)
        t1 = np.cos(pi*x)
        t2 = np.sin(pi*y)
        t3 = np.cos(pi*y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = m*t1*t2 +  x*(1-x)*(1-2*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val = (1-x)*y*(1-y) - x*y*(1-y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*(1-y) - x*(1-x)*y
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        t0 = np.sin(pi*x)
        t1 = np.cos(pi*x)
        t2 = np.sin(pi*y)
        t3 = np.cos(pi*y)
        val = np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        return val

class ExponentData:
    """
    u = (-y*x*(1-x),x*y*(1-y))^t
    p = np.exp(x*(1-x)*y*(1-y)) - 1
    g = x - y
    f = (-(mu/k + rho*beta*np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2)))*x*y*(1-x) + (1-2*x)*y*(1-y)*np.exp(x*(1-x)*y*(1-y)),
        (mu/k + rho*beta*np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2)))*x*y*(1-y) + x*(1-x)*(1-2*y)*np.exp(x*(1-x)*y*(1-y)))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh1(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = x-y
        return rhs

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = -y*x*(1-x)
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*y*(1-y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.exp(x*(1-x)*y*(1-y)) - 1
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = -(mu/k + rho*beta*np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2)))
        val = m*x*y*(1-x) + (1-2*x)*y*(1-y)*np.exp(x*(1-x)*y*(1-y))
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2))
        val = m*x*y*(1-y) + x*(1-x)*(1-2*y)*np.exp(x*(1-x)*y*(1-y))
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = (1-2*x)*y*(1-y)*np.exp(x*(1-x)*y*(1-y))
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*(1-2*y)*np.exp(x*(1-x)*y*(1-y))
        return val
    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sqrt(x**2*y**2*((1-x)**2 + (1-y)**2))
        return val

class SinsinData:
    """
    u = (np.sin(x)*np.cos(y),-np.cos(x)*np.sin(y))^t
    p = np.sin(pi*x)*np.sin(pi*y)
    g = 0
    f = ((mu/k + rho*beta*np.sqrt(np.sin(x)**2*np.cos(y)**2 + np.cos(x)**2*np.sin(y)**2))*np.sin(x)*np.cos(x)*np.sin(y) + pi*np.cos(pi*x)*np.sin(pi*y),
         -(mu/k + rho*beta*np.sqrt(np.sin(x)**2*np.cos(y)**2 + np.cos(x)**2*np.sin(y)**2))*np.cos(x)*np.sin(y)  + pi*np.sin(pi*x)*np.cos(pi*y))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1

    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p.shape[0]
        rhs = np.zeros(x,)
        return rhs

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sin(x)*np.cos(y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = -np.cos(x)*np.sin(y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = m*t0*t3 + pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        val = -m*t1*t2 + pi*np.sin(pi*x)*np.cos(pi*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val= pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = pi*np.sin(pi*x)*np.cos(pi*y)
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(y)
        t3 = np.cos(y)
        val = np.sqrt(t0**2*t3**2 + t1**2*t2**2)
        return val

class ArctanData:
    """
    u = (-y,x)^t
    p = 2/np.pi*np.arctan(10*(x+y-1))
    g = 0
    f = (-(mu/k + rho*beta*np.sqrt(x**2+y**2))*y + 20/(np.pi*(1+(10*(x+y-1))**2)),
         (mu/k + rho*beta*np.sqrt(x**2+y**2))*x + 20/(np.pi*(1+(10*(x+y-1))**2)))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1

    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh1(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        rhs = np.zeros(p.shape[0],)

        return rhs

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = -y
        return val

    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = 2/np.pi*np.arctan(10*(x+y-1))
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(x**2+y**2)
        val = -m*y + 20/(np.pi*(1+(10*(x+y-1))**2))
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(x**2 + y**2)
        val = m*x + 20/(np.pi*(1+(10*(x+y-1))**2))
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 20/(np.pi*(1+(10*(x+y-1))**2))
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 20/(np.pi*(1+(10*(x+y-1))**2))
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sqrt(x**2 + y**2)
        return val

class DeltaData:
    """
    g = 0
    f = pi*(delta(0,0)-delta(1,1))
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1

    def init_mesh(self, nx, ny):
        box = self.box
        mesh = StructureQuadMesh1(box, nx, ny)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        rhs = np.zeros(p.shape[0],)
        rhs[0] = np.pi
        rhs[-1] = -np.pi

        return rhs

    def source2(self, p):

        rhs = np.zeros(p.shape[0],)

        return rhs

    def source3(self, p):

        rhs = np.zeros(p.shape[0],)

        return rhs

    def velocity_x(self,p):
        val = np.zeros(p.shape[0],)
        return val

    def velocity_y(self, p):
        val = np.zeros(p.shape[0],)
        return val


class DarcyForchheimerdata1:
    """
    p = x^3 + y^3
    u = (x+y,x-y)^t
    g = 0
    f = ((1+beta*sqrt(2x^2+2y^2))(x+y)+3x^2,(1+beta*sqrt(2x^2+2y^2))(x-y)+3y^2)
    """
    def __init__(self, box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN):
        self.box = box
        self.mu = mu
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.level = level
        self.tol = tol
        self.maxN = maxN
        self.mg_maxN = mg_maxN

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)

        if meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)

    def g(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        rhs = np.zeros(p.shape[0:-1],)

        return rhs

    def velocity(self,p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = x + y
        val[..., 1] = x - y
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = x**3 + y**3
        return val
    def f(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)

        beta = self.beta
        m = np.sqrt(2*x**2+2*y**2)
        val[..., 0] = (1 + beta*m)*(x+y) + 3*x**2
        val[..., 1] = (1 + beta*m)*(x-y) + 3*y**2
        return val

    def grad_pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 3*x**2
        val[..., 1] = 3*y**2
        return val

    def neumann(self,p):
        z = np.zeros(p.shape[0],)
        x = p[..., 0]
        y = p[..., 1]
        idx = np.nonzero(abs(x - 1) < np.spacing(0))
        z[idx] = 1 + y[idx]
        
        idx = np.nonzero(abs(x + 1) < np.spacing(0))
        z[idx] = 1 - y[idx]
        
        idx = np.nonzero(abs(y - 1) < np.spacing(0))
        z[idx] = x[idx] - 1
        
        idx = np.nonzero(abs(y + 1) < np.spacing(0))
        z[idx] = - x[idx] - 1
        
        return z   
        
class Example7:
    """
    p = (x-x^2)(y-y^2)
    u = (exp(x)sin(y),exp(x)cos(y))^t
    g = exp(x)+exp(y)
    f = (mu/rho/k*sqrt((exp(x)sin(y))^2 + (exp(x)cos(y))^2))*u + ((1-2x)(y-y^2),(x-x^2)(1-2y))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = np.zeros(p.shape[0],)
        return rhs


    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:,0] = exp(x)*np.sin(y)
        val[:,1] = exp(x)*np.cos(y)
        return val

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.exp(x)*np.sin(y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.exp(x)*np.cos(y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*y*(1-y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(np.exp(x)**2*t2**2 + np.exp(x)**2*t3**2)
        val = m*np.exp(x)*t2 + (1-2*x)*y*(1-y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(y)
        t1 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(t0**2*np.exp(x)**2 + t1**2*np.exp(x)**2)
        val = m*t1*np.exp(x) +  x*(1-x)*(1-2*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val = (1-x)*y*(1-y) - x*y*(1-y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*(1-y) - x*(1-x)*y
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]

        t0 = np.sin(y)
        t1 = np.cos(y)
        val = np.sqrt(t0**2*np.exp(x)**2 + t1**2*np.exp(x)**2)
        return val
       
class Example8:
    """
    p = sin(pi*x)sin(pi*y)
    u = (exp(x)sin(y),exp(x)cos(y))^t
    g = exp(x)+exp(y)
    f = (mu/rho/k*sqrt((exp(x)sin(y))^2 + (exp(x)cos(y))^2))*u + (pi*cos(pi*x)sin(pi*y),pi*sin(pi*x)cos(pi*y))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = np.zeros(p.shape[0],)
        return rhs


    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:,0] = exp(x)*np.sin(y)
        val[:,1] = exp(x)*np.cos(y)
        return val

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = exp(x)*np.sin(y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = exp(x)*np.cos(y)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sin(np.pi*x)*np.sin(np.pi*y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(exp(x)**2*t2**2 + exp(x)**2*t3**2)
        val = m*exp(x)*t2 + np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(y)
        t1 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(t0**2*exp(x)**2 + t1**2*exp(x)**2)
        val = m*t1*exp(x) +  np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]

        t0 = np.sin(y)
        t1 = np.cos(y)
        val = np.sqrt(t0**2*exp(x)**2 + t1**2*exp(x)**2)
        return val
      
 
class Example9:
    """
    p = (x-x^2)(y-y^2)
    u = (x*exp(y),y*exp(x))^t
    g = exp(x)+exp(y)
    f = (mu/rho/k*sqrt((x*exp(y))^2 + (y*exp(x))^2))*u + ((1-2x)(y-y^2),(x-x^2)(1-2y))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        rhs = exp(x) + exp(y)
        return rhs


    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:,0] = x*exp(y)
        val[:,1] = y*exp(x)
        return val

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = x*exp(y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = y*exp(x)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*y*(1-y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        m = mu/k + rho*beta*np.sqrt(exp(y)**2*x**2 + exp(x)**2*y**2)
        val = m*exp(x)*t2 + (1-2*x)*y*(1-y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(y)
        t1 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(exp(y)**2*x**2 + exp(x)**2*y**2)
        val = m*t1*exp(x) +  x*(1-x)*(1-2*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val = (1-x)*y*(1-y) - x*y*(1-y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = x*(1-x)*(1-y) - x*(1-x)*y
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]

        t0 = np.sin(y)
        t1 = np.cos(y)
        val = np.sqrt(exp(y)**2*x**2 + exp(x)**2*y**2)
        return val
        
class Example10:
    """
    p = sin(pi*x)sin(pi*y)
    u = (x*exp(y),y*exp(x))^t
    g = exp(x)+exp(y)
    f = (mu/rho/k*sqrt((x*exp(y))^2 + (y*exp(x))^2))*u + (pi*cos(pi*x)sin(pi*y),pi*sin(pi*x)cos(pi*y))^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = exp(x) + exp(y)
        return rhs


    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[:,0] = x*exp(y)
        val[:,1] = y*exp(x)
        return val

    def velocity_x(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = x*exp(y)
        return val
    def velocity_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = y*exp(x)
        return val

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.sin(np.pi*x)*np.sin(np.pi*y)
        return val
    def source2(self, p):
        x = p[..., 0]
        y = p[..., 1]
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t2 = np.sin(y)
        t3 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(exp(y)**2*x**2 + exp(x)**2*y**2)
        val = m*exp(x)*t2 + np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        return val

    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        mu = self.mu
        k = self.k
        rho = self.rho
        beta = self.beta
        t0 = np.sin(y)
        t1 = np.cos(y)
        m = mu/k + rho*beta*np.sqrt(exp(y)**2*x**2 + exp(x)**2*y**2)
        val = m*t1*exp(x) +  np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        return val
    def grad_pressure_x(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape)
        val = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        return val

    def grad_pressure_y(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        return val

    def normu(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.sqrt(exp(y)**2*x**2 + exp(x)**2*y**2)
        return val
        
        
class Example11:
    """
    g = 0
    f = (sin(pi*x), sin(pi*y))
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.zeros(p.shape[0],)
        return rhs
        
    def source2(sel, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.sin(np.pi*x)
        return rhs
        
    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.sin(np.pi*y)
        return rhs
        
    def velocity_x(self,p):
        val = np.zeros(p.shape[0],)
        return val

    def velocity_y(self, p):
        val = np.zeros(p.shape[0],)
        return val
        
class Example12:
    """
    g = 0
    f = (x-x^2, y-y^2)
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.zeros(p.shape[0],)
        return rhs
        
    def source2(sel, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = x - x**2
        return rhs
        
    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = y - y**2
        return rhs
        
    def velocity_x(self,p):
        val = np.zeros(p.shape[0],)
        return val

    def velocity_y(self, p):
        val = np.zeros(p.shape[0],)
        return val

class Example13:
    """
    g = exp(-((x - 1/2)^2 + (y - 1/2)^2))
    f = (1,0)^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.exp(-((x-1/2)**2 + (y-1/2)**2))
        return rhs
        
    def source2(sel, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.ones(p.shape[0],)
        return rhs
        
    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.zeros(p.shape[1],)
        return rhs
        
    def velocity_x(self,p):
        val = np.zeros(p.shape[0],)
        return val

    def velocity_y(self, p):
        val = np.zeros(p.shape[0],)
        return val
        
class Example14:
    """
    g = sin(pi*x)sin(2*pi*y)
    f = (1,0)^t
    """
    def __init__(self, box, mu, k, rho, beta, tol):
        self.box = box
        self.mu = mu
        self.k = k
        self.rho = rho
        self.beta = beta
        self.tol = tol
        self.flat = 1


    def init_mesh(self, hx, hy):
        box = self.box
        mesh = StructureQuadMesh1(box, hx, hy)
        return mesh


    def source1(self, p):
        """ The right hand side of DarcyForchheimer equation
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.sin(np.pi*x)*np.sin(2*np.pi*y)
        return rhs
        
    def source2(sel, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.ones(p.shape[0],)
        return rhs
        
    def source3(self, p):
        x = p[..., 0]
        y = p[..., 1]
        rhs = np.zeros(p.shape[1],)
        return rhs
        
    def velocity_x(self,p):
        val = np.zeros(p.shape[0],)
        return val

    def velocity_y(self, p):
        val = np.zeros(p.shape[0],)
        return val

       
              
