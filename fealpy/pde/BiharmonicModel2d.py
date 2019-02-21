import numpy as np
from ..mesh.TriangleMesh import TriangleMesh  


class VSSPData:
    def __init__(self):
        pass

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = np.sin(pi*x)*np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*y)
        return r

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        return val

    def init_mesh(self, meshtype='tri', n=1, ftype=np.float, itype=np.int):
        node = np.array([
            (0.0, 0.0),
            (0.5, 0.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0, 0.5),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0)], dtype=ftype)
        if meshtype is 'tri':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=itype)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'quadtree':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=itype)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)

class SinSinData:
    def __init__(self):
        pass

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = np.sin(pi*x)*np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*y)
        return r

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[...,0] = 2*pi*np.sin(pi*x)*np.cos(pi*x)*np.sin(pi*y)*np.sin(pi*y)
        val[...,1] = 2*pi*np.sin(pi*x)*np.sin(pi*x)*np.sin(pi*y)*np.cos(pi*y)
        return val


    def laplace(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = 2*pi**2*np.cos(pi*y)**2*np.sin(pi*x)**2
        r += 2*pi**2*np.cos(pi*x)**2*np.sin(pi*y)**2 
        r -= 4*pi**2*np.sin(pi*x)**2*np.sin(pi*y)**2
        return r

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def laplace_dirichlet(self, p):
        return self.laplace(p);

    def laplace_neuman(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 4*pi**3*(-sin(pi*y)**2 + cos(pi*y)**2)*sin(pi*x)*cos(pi*x) - 8*pi**3*sin(pi*x)*sin(pi*y)**2*cos(pi*x)
        val[..., 1] = 4*pi**3*(-sin(pi*x)**2 + cos(pi*x)**2)*sin(pi*y)*cos(pi*y) - 8*pi**3*sin(pi*x)**2*sin(pi*y)*cos(pi*y)
        return np.sum(val*n, axis=-1) 

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        pi4 = pi**4
        r1 = np.sin(pi*x)**2
        r2 = np.cos(pi*x)**2
        r3 = np.sin(pi*y)**2
        r4 = np.cos(pi*y)**2
        r = 8*pi4*r2*r4 - 16*pi4*r4*r1 - 16*pi4*r2*r3 + 24*pi4*r1*r3
        return r

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

class BiharmonicData2:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def solution(self,p):
        """ The exact solution 
        """
        a = self.a
        b = self.b
        x = p[..., 0]
        y = p[..., 1]
        r = 2350*(x**4)*(x-a)*(x-a)*(y**4)*(y-b)*(y-b)
        return r

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        b = self.b
        val = np.zeros(p.shape, dtype=p.dtype)
        val[...,0] = 2350*2*(x**3)*(x-a)*(3*x-2*a)*(y**4)*(y-b)*(y-b)
        val[...,1] = 2350*(x**4)*(x-a)*(x-a)*2*(y**3)*(y-b)*(3*y-2*b)
        return val

    def laplace(self,p):
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        b = self.b
        r = 2350*(y**6-2*b*y**5+b**2*y**4)*(30*x**4-40*a*x**3+12*a**2*x**2)
        r += 2350*(x**6-2*a*x**5+a**2*x**4)*(30*y**4-40*b*y**3+12*b**2*y**2)
        return r


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return np.zeros((p.shape[0],), dtype=np.float)

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        return np.zeros(p.shape[0:-1], dtype=np.float)

    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        b = self.b
        r1 = 56400*(a**2-10*a*x+15*x**2)*(b-y)*(b-y)*y**4
        r2 = 18800*x**2*(6*a**2-20*a*x+15*x**2)*y**2*(6*b**2-20*b*y+15*y**2)
        r3 = 56400*(a-x)*(a-x)*x**4*(b**2-10*b*y+15*y**2)
        r = r1 + r2 + r3
        return r

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

class BiharmonicData3:
    def __init__(self):
        pass
    
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = np.cos(2*pi*x)*np.cos(2*pi*y)
        return r

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros((len(x), 2), dtype=p.dtype)
        val[...,0] = -2*pi*np.sin(2*pi*x)*np.cos(2*pi*y) 
        val[...,1] = -2*pi*np.cos(2*pi*x)*np.sin(2*pi*y)
        return val

    def laplace(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = -8*pi**2*self.solution(p)
        return r


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p) 

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        return np.zeros(p.shape[0], dtype=p.dtype) 

    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = 64*pi**4*np.cos(2*pi*x)*np.cos(2*pi*y)
        return r

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

class BiharmonicData4:
    def __init__(self):
        pass
    
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = np.sin(2*pi*x)*np.sin(2*pi*y)
        return r

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[...,0] = 2*pi*np.cos(2*pi*x)*np.sin(2*pi*y) 
        val[...,1] = 2*pi*np.cos(2*pi*y)*np.sin(2*pi*x)
        return val


    def laplace(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = -8*pi**2*self.solution(p)
        return r

    def laplace_dirichlet(self, p):
        return self.laplace(p);

    def laplace_neuman(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -16*pi**3*sin(2*pi*y)*cos(2*pi*x)
        val[..., 1] = -16*pi**3*sin(2*pi*x)*cos(2*pi*y)
        return np.sum(val*n, axis=-1) 


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p) 

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        r = 64*pi**4*np.sin(2*pi*x)*np.sin(2*pi*y)
        return r


class BiharmonicData5:
    def __init__(self, a=0.1):
        self.a = a

    def init_mesh(self, n=5):
        point = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)
        cell = np.array([
            (1, 2, 0),
            (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n=n)
        return mesh 

    def domain(self):
        point = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)
        segment = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0)], dtype=np.float)
        return point, segment
    
    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        r1 = (x - 0.5)**2 + (y - 0.5)**2 + a 
        r2 = (x + 0.5)**2 + (y + 0.5)**2 + a 
        return 1.0/r1 - 1.0/r2 

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -(-2*x - 1.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**2 + (-2*x + 1.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**2
        val[..., 1] = -(-2*y - 1.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**2 + (-2*y + 1.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**2
        return val


    def laplace(self,p):
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        r =(2*x - 1.0)*(4*x - 2.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**3 - (2*x + 1.0)*(4*x + 2.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**3 + (2*y - 1.0)*(4*y - 2.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**3 - (2*y + 1.0)*(4*y + 2.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**3 + 4/(a + (x + 0.5)**2 + (y + 0.5)**2)**2 - 4/(a + (x - 0.5)**2 + (y - 0.5)**2)**2 
        return r


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        a = self.a
        r = (2*x - 1.0)*(4*x - 2.0)*(6*x - 3.0)*(8*x - 4.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**5 + 2*(2*x - 1.0)*(4*x - 2.0)*(6*y - 3.0)*(8*y - 4.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**5 - 18*(2*x - 1.0)*(4*x - 2.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**4 - 8*(2*x - 1.0)*(6*x - 3.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**4 - (2*x + 1.0)*(4*x + 2.0)*(6*x + 3.0)*(8*x + 4.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**5 - 2*(2*x + 1.0)*(4*x + 2.0)*(6*y + 3.0)*(8*y + 4.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**5 + 18*(2*x + 1.0)*(4*x + 2.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**4 + 8*(2*x + 1.0)*(6*x + 3.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**4 - 6*(4*x - 2.0)*(6*x - 3.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**4 + 6*(4*x + 2.0)*(6*x + 3.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**4 + (2*y - 1.0)*(4*y - 2.0)*(6*y - 3.0)*(8*y - 4.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**5 - 6*(2*y - 1.0)*(4*y - 2.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**4 - 8*(2*y - 1.0)*(6*y - 3.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**4 - (2*y + 1.0)*(4*y + 2.0)*(6*y + 3.0)*(8*y + 4.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**5 + 6*(2*y + 1.0)*(4*y + 2.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**4 + 8*(2*y + 1.0)*(6*y + 3.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**4 - 10*(4*y - 2.0)*(6*y - 3.0)/(a + (x - 0.5)**2 + (y - 0.5)**2)**4 + 10*(4*y + 2.0)*(6*y + 3.0)/(a + (x + 0.5)**2 + (y + 0.5)**2)**4 - 64/(a + (x + 0.5)**2 + (y + 0.5)**2)**3 + 64/(a + (x - 0.5)**2 + (y - 0.5)**2)**3
        return r


class BiharmonicData6:
    def __init__(self):
        pass

    def init_mesh(self, n=1):
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
            (1, 3, 0),
            (2, 0, 3),
            (3, 6, 2),
            (5, 2, 6),
            (4, 7, 3),
            (6, 3, 7)], dtype=np.int)
        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n=n)
        return mesh 

    def domain(self):
        vertex = np.array([
            (-1, -1),
            (0, -1),
            (0, 0),
            (1, 0),
            (-1, 1),
            (1, 1)], dtype=np.float)
        segment = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0)], dtype=np.float)
        return point, segment

    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x*x + y*y
        z = r**(5/6)*np.sin(5*theta/3)
        return z

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        sin = np.sin
        cos = np.cos
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x*x + y*y
        val = np.zeros((len(x), 2), dtype=p.dtype)
        val[..., 0] = 5*x*sin(5*theta/3)/(3*r**(1/6)) - 5*y*cos(5*theta/3)/(3*r**(1/6))
        val[..., 1] = 5*x*cos(5*theta/3)/(3*r**(1/6)) + 5*y*sin(5*theta/3)/(3*r**(1/6))
        return val

    def laplace(self,p):
        z = np.zeros(p.shape[0:-1], dtype=np.float)
        return z


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self,p):
        z = np.zeros(p.shape[0:-1], dtype=np.float)
        return z

class BiharmonicData7:
    def __init__(self):
        self.z = 0.544483736782464

    def init_mesh(self, n=1):
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
            (1, 3, 0),
            (2, 0, 3),
            (3, 6, 2),
            (5, 2, 6),
            (4, 7, 3),
            (6, 3, 7)], dtype=np.int)
        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n=n)
        return mesh 

    def domain(self):
        point = np.array([
            (-1, -1),
            (0, -1),
            (0, 0),
            (1, 0),
            (1, 1),
            (-1, 1)], dtype=np.float)
        segment = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0)], dtype=np.float)
        return point, segment

    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        d = np.sqrt(1 - z**2)
        val = -4*z*(x**2 - 1)**2*r**z*(y**2 - 1)**2*(r*z + s*y)/d
        return val

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        d = r**3*np.sqrt(1 - z**2)
        m1 = 4*z*(r**z)*(x**2 - 1)*(y**2 - 1)**2 
        m2 = -4*z*(r**z)*(x**2 - 1)**2*(y**2 - 1) 
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = m1*(c*r*y**2*z*(x**2 - 1) - x**5*z - x**3*z*(r**2*z + r*s*y + y**2 - 1) + x*(-4*r**4*z - 4*r**3*s*y + r**2*z**2 + r*s*y*z + y**2*z))/d
        val[..., 1] = m2*(c*r*x*y*z*(y**2 - 1) + r**2*y*z*(4*r**2 - z) + r*s*y**2*(5*r**2 - z) - r*s*(r**2 - y**4*z) + x**2*y*z*(y**2 - 1) + y**5*z + y**3*z*(r**2*z - 1))/d 
        return val

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        val = np.zeros(p.shape[:-1]+(2,2), dtype=p.dtype)


    def laplace(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        d = r**3*np.sqrt(1 - z**2)
        z1 = (z+1)**2
        z2 = (z+3)**2
        y1 = (5*y**2 - 1)*(y**2 - 1)
        y2 = (y - 1)*(y + 1)*(2*y**4 + 3*y**2 - 1)
        y3 = (y - 1)*(y + 1)*(4*y**4 + y**2 - 1)
        val = -4*z*r**z*(2*c*r*x**5*z*y1 - 4*c*r*x**3*z*y2 + 2*c*r*x*z*y3 + 2*r*s*y**5*(5*z + 14) - 4*r*s*y**3*(3*z + 4) - 2*r*s*y*(2*y**6 - z) + 4*x**8*z*(3*y**2 - 1) + x**6*(20*r*s*y**3 - 12*r*s*y + y**4*z**3 + 18*y**4*z**2 + 53*y**4*z - 2*y**2*z**3 - 28*y**2*z**2 - 82*y**2*z + z**3 + 10*z**2 + 29*z) + x**4*(18*r*s*y**5*z + 32*r*s*y**5 - 28*r*s*y**3*z - 76*r*s*y**3 + 10*r*s*y*z + 36*r*s*y + y**6*z**3 + 18*y**6*z**2 + 53*y**6*z - 4*y**4*z**3 - 56*y**4*z**2 - 156*y**4*z + 5*y**2*z**3 + 50*y**2*z**2 + 105*y**2*z - 2*z**3 - 12*z**2 - 18*z) + x**2*(12*r*s*y**7 - 28*r*s*y**5*z - 68*r*s*y**5 + 40*r*s*y**3*z + 64*r*s*y**3 - 12*r*s*y*z - 16*r*s*y + 12*y**8*z - 2*y**6*z**3 - 28*y**6*z**2 - 82*y**6*z + 5*y**4*z**3 + 50*y**4*z**2 + 105*y**4*z - 4*y**2*z**3 - 24*y**2*z**2 - 36*y**2*z + z**3 + 2*z**2 + z) - 4*y**8*z + y**6*z*(z**2 + 10*z + 29) - 2*y**4*z*z2 + y**2*z*z1)/(d)
        return val 


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        d = r**7*np.sqrt(1-z**2)
        y1 = 30*r**2*y**2 - 6*r**2 - 15*y**4*z + 20*y**4 + 18*y**2*z - 24*y**2 - 3*z + 4
        y2 = (30*r**4*y**2 - 6*r**4 + 95*r**2*y**4*z - 80*r**2*y**4 - 90*r**2*y**2*z + 18*r**2*y**2 + 11*r**2*z + 6*r**2 - 3*y**6*z + 4*y**6 + 24*y**4*z - 32*y**4 - 27*y**2*z + 36*y**2 + 6*z - 8)
        y3 = 10*r**4*y**4 + 72*r**4*y**2 - 18*r**4 + 76*r**2*y**6*z - 58*r**2*y**6 + 30*r**2*y**4*z - 14*r**2*y**4 - 88*r**2*y**2*z + 72*r**2*y**2 + 14*r**2*z - 8*r**2 - 12*y**8*z + 16*y**8 + 6*y**6*z - 8*y**6 + 15*y**4*z - 20*y**4 - 12*y**2*z + 16*y**2 + 3*z - 4
        y4 = 12*r**4*y**6 - 54*r**4*y**4 - 6*r**4*y**2 + 8*r**4 + 12*r**2*y**8 - 60*r**2*y**6*z + 10*r**2*y**6 + 33*r**2*y**4*z - 42*r**2*y**4 + 14*r**2*y**2*z - 8*r**2*y**2 - 3*r**2*z + 4*r**2 + 12*y**8*z - 16*y**8 - 9*y**6*z + 12*y**6 - 6*y**4*z + 8*y**4 + 3*y**2*z - 4*y**2
        val = -4*z*r**z*(4*c*r*x**7*z*y1 + 4*c*r*x**5*z*y2 - 4*c*r*x**3*z*y3 - 4*c*r*x*z*y4 + 4*r**4*z**3 + 8*r**4*z**2 + 4*r**4*z - 4*r**3*s*y*(24*r**2*z + 32*r**2 - 3*z**2 + 4*z) - 2*r**2*y**2*z*(16*r**2*z**2 + 96*r**2*z + 144*r**2 - 3*z**3 + 3*z**2 + 15*z + 9) + 8*r*s*y**9*(3*z - 4) + 4*r*s*y**7*(6*r**4 + 4*r**2*z**2 - 16*r**2*z + 32*r**2 - 15*z**2 - 22*z + 56) - 4*r*s*y**5*(14*r**4*z + 76*r**4 - 27*r**2*z**2 - 24*r**2*z + 168*r**2 - 18*z**2 + 32) + 4*r*s*y**3*(70*r**4*z + 172*r**4 - 26*r**2*z**2 + 16*r**2*z + 64*r**2 - 3*z**2 + 4*z) + 4*x**10*z*(3*y**2*z**2 - 24*y**2*z + 45*y**2 - z**2 + 8*z - 15) + x**8*(24*r**4*z + 264*r**2*y**2*z**2 - 792*r**2*y**2*z - 72*r**2*z**2 + 216*r**2*z - 120*r*s*y**3*z + 160*r*s*y**3 + 72*r*s*y*z - 96*r*s*y + y**4*z**5 + 10*y**4*z**4 - 64*y**4*z**3 - 250*y**4*z**2 + 975*y**4*z - 2*y**2*z**5 - 12*y**2*z**4 + 108*y**2*z**3 + 268*y**2*z**2 - 1290*y**2*z + z**5 + 2*z**4 - 36*z**3 - 82*z**2 + 435*z) + 2*x**6*(60*r**5*s*y + 6*r**4*y**2*z**3 + 108*r**4*y**2*z**2 + 654*r**4*y**2*z - 2*r**4*z**3 - 28*r**4*z**2 - 194*r**4*z - 40*r**3*s*y**3*z**2 + 200*r**3*s*y**3*z - 400*r**3*s*y**3 + 24*r**3*s*y*z**2 - 96*r**3*s*y*z + 192*r**3*s*y + 11*r**2*y**4*z**4 + 165*r**2*y**4*z**3 - 11*r**2*y**4*z**2 - 1749*r**2*y**4*z - 18*r**2*y**2*z**4 - 198*r**2*y**2*z**3 + 18*r**2*y**2*z**2 + 2214*r**2*y**2*z + 7*r**2*z**4 + 49*r**2*z**3 - 7*r**2*z**2 - 609*r**2*z - 54*r*s*y**5*z**2 - 84*r*s*y**5*z + 208*r*s*y**5 + 84*r*s*y**3*z**2 + 152*r*s*y**3*z - 352*r*s*y**3 - 30*r*s*y*z**2 - 68*r*s*y*z + 144*r*s*y + y**6*z**5 + 10*y**6*z**4 - 76*y**6*z**3 - 154*y**6*z**2 + 795*y**6*z - 3*y**4*z**5 - 18*y**4*z**4 + 172*y**4*z**3 + 322*y**4*z**2 - 1785*y**4*z + 3*y**2*z**5 + 6*y**2*z**4 - 128*y**2*z**3 - 86*y**2*z**2 + 1005*y**2*z - z**5 + 2*z**4 + 24*z**3 - 18*z**2 - 135*z) + x**4*(360*r**5*s*y**3*z + 1240*r**5*s*y**3 - 168*r**5*s*y*z - 816*r**5*s*y + 60*r**4*y**4*z**3 + 1080*r**4*y**4*z**2 + 3180*r**4*y**4*z - 108*r**4*y**2*z**3 - 1512*r**4*y**2*z**2 - 4332*r**4*y**2*z + 40*r**4*z**3 + 400*r**4*z**2 + 1080*r**4*z + 556*r**3*s*y**5*z**2 - 80*r**3*s*y**5*z - 1280*r**3*s*y**5 - 536*r**3*s*y**3*z**2 - 320*r**3*s*y**3*z + 2432*r**3*s*y**3 + 76*r**3*s*y*z**2 + 192*r**3*s*y*z - 864*r**3*s*y + 22*r**2*y**6*z**4 + 330*r**2*y**6*z**3 - 22*r**2*y**6*z**2 - 3498*r**2*y**6*z - 72*r**2*y**4*z**4 - 792*r**2*y**4*z**3 + 216*r**2*y**4*z**2 + 8424*r**2*y**4*z + 70*r**2*y**2*z**4 + 490*r**2*y**2*z**3 - 630*r**2*y**2*z**2 - 4410*r**2*y**2*z - 20*r**2*z**4 - 60*r**2*z**3 + 180*r**2*z**2 + 540*r**2*z - 108*r*s*y**7*z**2 - 120*r*s*y**7*z + 352*r*s*y**7 + 336*r*s*y**5*z**2 + 416*r*s*y**5*z - 1152*r*s*y**5 - 300*r*s*y**3*z**2 - 200*r*s*y**3*z + 800*r*s*y**3 + 72*r*s*y*z**2 - 128*r*s*y + y**8*z**5 + 10*y**8*z**4 - 64*y**8*z**3 - 250*y**8*z**2 + 975*y**8*z - 6*y**6*z**5 - 36*y**6*z**4 + 344*y**6*z**3 + 644*y**6*z**2 - 3570*y**6*z + 10*y**4*z**5 + 20*y**4*z**4 - 440*y**4*z**3 - 180*y**4*z**2 + 3150*y**4*z - 6*y**2*z**5 + 12*y**2*z**4 + 144*y**2*z**3 - 108*y**2*z**2 - 810*y**2*z + z**5 - 6*z**4 + 22*z**2 + 15*z) + 2*x**2*(108*r**5*s*y**5*z + 444*r**5*s*y**5 - 448*r**5*s*y**3*z - 1136*r**5*s*y**3 + 180*r**5*s*y*z + 408*r**5*s*y + 6*r**4*y**6*z**3 + 108*r**4*y**6*z**2 + 654*r**4*y**6*z - 54*r**4*y**4*z**3 - 756*r**4*y**4*z**2 - 2166*r**4*y**4*z + 60*r**4*y**2*z**3 + 600*r**4*y**2*z**2 + 1260*r**4*y**2*z - 16*r**4*z**3 - 96*r**4*z**2 - 144*r**4*z - 24*r**3*s*y**7*z**2 + 120*r**3*s*y**7*z - 240*r**3*s*y**7 - 284*r**3*s*y**5*z**2 - 96*r**3*s*y**5*z + 1088*r**3*s*y**5 + 312*r**3*s*y**3*z**2 - 96*r**3*s*y**3*z - 768*r**3*s*y**3 - 52*r**3*s*y*z**2 + 32*r**3*s*y*z + 128*r**3*s*y + 132*r**2*y**8*z**2 - 396*r**2*y**8*z - 18*r**2*y**6*z**4 - 198*r**2*y**6*z**3 + 18*r**2*y**6*z**2 + 2214*r**2*y**6*z + 35*r**2*y**4*z**4 + 245*r**2*y**4*z**3 - 315*r**2*y**4*z**2 - 2205*r**2*y**4*z - 20*r**2*y**2*z**4 - 60*r**2*y**2*z**3 + 180*r**2*y**2*z**2 + 540*r**2*y**2*z + 3*r**2*z**4 - 3*r**2*z**3 - 15*r**2*z**2 - 9*r**2*z - 36*r*s*y**9*z + 48*r*s*y**9 + 84*r*s*y**7*z**2 + 104*r*s*y**7*z - 288*r*s*y**7 - 150*r*s*y**5*z**2 - 76*r*s*y**5*z + 368*r*s*y**5 + 72*r*s*y**3*z**2 - 128*r*s*y**3 - 6*r*s*y*z**2 + 8*r*s*y*z + 6*y**10*z**3 - 48*y**10*z**2 + 90*y**10*z - y**8*z**5 - 6*y**8*z**4 + 54*y**8*z**3 + 134*y**8*z**2 - 645*y**8*z + 3*y**6*z**5 + 6*y**6*z**4 - 128*y**6*z**3 - 86*y**6*z**2 + 1005*y**6*z - 3*y**4*z**5 + 6*y**4*z**4 + 72*y**4*z**3 - 54*y**4*z**2 - 405*y**4*z + y**2*z**5 - 6*y**2*z**4 + 22*y**2*z**2 + 15*y**2*z) - 4*y**10*z*(z**2 - 8*z + 15) + y**8*z*(24*r**4 - 72*r**2*z + 216*r**2 + z**4 + 2*z**3 - 36*z**2 - 82*z + 435) - 2*y**6*z*(2*r**4*z**2 + 28*r**4*z + 194*r**4 - 7*r**2*z**3 - 49*r**2*z**2 + 7*r**2*z + 609*r**2 + z**4 - 2*z**3 - 24*z**2 + 18*z + 135) + y**4*z*(40*r**4*z**2 + 400*r**4*z + 1080*r**4 - 20*r**2*z**3 - 60*r**2*z**2 + 180*r**2*z + 540*r**2 + z**4 - 6*z**3 + 22*z + 15))/d 
        return val 



class BiharmonicData8:
    def __init__(self):
        self.z = 0.505009698896589

    def init_mesh(self, n=1):
        point = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1), 
            (1, -1)], dtype=np.float)
        cell = np.array([
            (1, 3, 0),
            (2, 0, 3),
            (3, 6, 2),
            (5, 2, 6),
            (4, 7, 3),
            (6, 3, 7),
            (1, 8, 3)], dtype=np.int)
        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n=n)
        return mesh 

    def domain(self):
        point = np.array([
            (-1, -1),
            (1, -1),
            (0, 0),
            (1, 0),
            (1, 1),
            (-1, 1)], dtype=np.float)
        segment = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0)], dtype=np.float)
        return point, segment

    def solution(self,p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        w = np.sin(7/4*pi)
        zw = np.sin(z*w)
        zw1 = (z*np.sqrt(-zw**2 + 1) + zw)
        s2 = np.sqrt(2)
        xy2 = ((x**2 - 1)*(y**2 - 1))**2
        val = r**z*s2*zw1*xy2*(r*s2*zw + 2*s*y)/((z - 1)*(z + 1))
        return val

    def gradient(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        w = np.sin(7/4*pi)
        zw = np.sin(z*w)
        zw1 = (z*np.sqrt(-zw**2 + 1) + zw)
        s2 = np.sqrt(2)
        xy2 = (x**2 - 1)*(y**2 - 1)**2
        val = np.zeros(p.shape, dtype=p.dtype)
        x3 = r*s2*z*zw + r*s2*zw + 2*s*y*z
        x1 = -4*r**3*s2*zw - 8*r**2*s*y + r*s2*z*zw + r*s2*zw + 2*s*y*z
        val[..., 0] = -r**(z - 2)*s2*zw1*xy2*(c*s2**2*y**2*z*(x**2 - 1) - x**3*x3 + x*x1)/((z - 1)*(z + 1))
        xy2 = (x**2 - 1)**2*(y**2 - 1)
        y2 = r**2*s2**2 + 8*r**2 - 2*z
        k = r*s2*y**3*zw*(z + 1) - r*s2*y*zw*(-4*r**2 + z + 1) + 2*s*y**4*z + s*y**2*y2
        val[..., 1] =r**(z - 2)*s2*zw1*xy2*(c*s2**2*x*y*z*(y**2 - 1) - r**2*s*s2**2 + k )/((z - 1)*(z + 1))
        return val

    def laplace(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        w = np.sin(7/4*pi)
        zw = np.sin(z*w)
        zw1 = (z*np.sqrt(-zw**2 + 1) + zw)
        s2 = np.sqrt(2)
        x1 = 2*s2**2*y**6 - s2**2*y**4 - 2*s2**2*y**2 + s2**2 + 4*y**6 - 4*y**4
        x2 = -12*r**5*s2*y**4*zw + 48*r**5*s2*y**2*zw - 20*r**5*s2*zw + 8*r**4*s*s2**2*y**3 - 8*r**4*s*s2**2*y - 24*r**4*s*y**5 + 112*r**4*s*y**3 - 56*r**4*s*y + 28*r**3*s2*y**4*z*zw + 28*r**3*s2*y**4*zw - 40*r**3*s2*y**2*z*zw - 40*r**3*s2*y**2*zw + 12*r**3*s2*z*zw + 12*r**3*s2*zw + 2*r**2*s*s2**2*y**5*z - 4*r**2*s*s2**2*y**3*z + 2*r**2*s*s2**2*y*z + 60*r**2*s*y**5*z - 88*r**2*s*y**3*z + 28*r**2*s*y*z + 2*r*s2*y**6*z**2*zw - 2*r*s2*y**6*zw - 5*r*s2*y**4*z**2*zw + 5*r*s2*y**4*zw + 4*r*s2*y**2*z**2*zw - 4*r*s2*y**2*zw - r*s2*z**2*zw + r*s2*zw - 2*s*s2**2*y**7*z**2 + 5*s*s2**2*y**5*z**2 - 4*s*s2**2*y**3*z**2 + s*s2**2*y*z**2 + 4*s*y**7*z**2 - 8*s*y**7*z - 10*s*y**5*z**2 + 20*s*y**5*z + 8*s*y**3*z**2 - 16*s*y**3*z - 2*s*y*z**2 + 4*s*y*z
        x3 = s2**2*y**6 + s2**2*y**4 - 3*s2**2*y**2 + s2**2 + 2*y**6 - 2*y**2
        x4 = 12*r**5*s2*y**2*zw - 4*r**5*s2*zw + 4*r**4*s*s2**2*y**3 - 4*r**4*s*s2**2*y + 32*r**4*s*y**3 - 16*r**4*s*y + 18*r**3*s2*y**4*z*zw + 18*r**3*s2*y**4*zw - 28*r**3*s2*y**2*z*zw - 28*r**3*s2*y**2*zw + 10*r**3*s2*z*zw + 10*r**3*s2*zw + r**2*s*s2**2*y**5*z - 2*r**2*s*s2**2*y**3*z + r**2*s*s2**2*y*z + 38*r**2*s*y**5*z - 60*r**2*s*y**3*z + 22*r**2*s*y*z + r*s2*y**6*z**2*zw - r*s2*y**6*zw - 4*r*s2*y**4*z**2*zw + 4*r*s2*y**4*zw + 5*r*s2*y**2*z**2*zw - 5*r*s2*y**2*zw - 2*r*s2*z**2*zw + 2*r*s2*zw - s*s2**2*y**7*z**2 + 4*s*s2**2*y**5*z**2 - 5*s*s2**2*y**3*z**2 + 2*s*s2**2*y*z**2 + 2*s*y**7*z**2 - 4*s*y**7*z - 8*s*y**5*z**2 + 16*s*y**5*z + 10*s*y**3*z**2 - 20*s*y**3*z - 4*s*y*z**2 + 8*s*y*z
        x5 = 3*s2**2*y**4 - 4*s2**2*y**2 + s2**2 + 4*y**4 - 4*y**2
        x6 = -r*s2*y**4*z**2*zw + r*s2*y**4*zw + 2*r*s2*y**2*z**2*zw - 2*r*s2*y**2*zw - r*s2*z**2*zw + r*s2*zw + s*s2**2*y**5*z**2 - 2*s*s2**2*y**3*z**2 + s*s2**2*y*z**2 - 2*s*y**5*z**2 + 4*s*y**5*z + 4*s*y**3*z**2 - 8*s*y**3*z - 2*s*y*z**2 + 4*s*y*z
        val =  r**(z - 4)*s2*zw1*(2*c*r**2*x**5*z*(x5) - 4*c*r**2*x**3*z*(x3) + 2*c*r**2*x*z*(x1) - 8*r**5*s2*zw + 2*r**3*s2*z*zw + 2*r**3*s2*zw - r**2*s*y*(4*r**2*s2**2 + 24*r**2 - s2**2*z - 6*z) + r*s2*y**6*zw*(z**2 - 1) + 2*r*s2*y**4*zw*(-2*r**4 + 5*r**2*z + 5*r**2 - z**2 + 1) - r*s2*y**2*zw*(-20*r**4 + 12*r**2*z + 12*r**2 - z**2 + 1) - s*y**7*z*(s2**2*z - 2*z + 4) + s*y**5*(-8*r**4 + r**2*s2**2*z + 22*r**2*z + 2*s2**2*z**2 - 4*z**2 + 8*z) - s*y**3*(-4*r**4*s2**2 - 48*r**4 + 2*r**2*s2**2*z + 28*r**2*z + s2**2*z**2 - 2*z**2 + 4*z) - x**6*(x6) + x**4*(x4) - x**2*(x2))/((z - 1)*(z + 1))
        return val 


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=1)

    def source(self,p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        z = self.z
        s = np.sin(z*theta)
        c = np.cos(z*theta)
        r = np.sqrt(x*x + y*y)
        w = np.sin(7/4*pi)
        zw = np.sin(z*w)
        zw1 = (z*np.sqrt(-zw**2 + 1) + zw)
        s2 = np.sqrt(2)
        x1 = 12*r**4*s2**2*y**6 - 64*r**4*s2**2*y**4 - 20*r**4*s2**2*y**2 + 20*r**4*s2**2 + 72*r**4*y**6 - 304*r**4*y**4 - 8*r**4*y**2 + 24*r**4 - 36*r**2*s2**2*y**6*z + 64*r**2*s2**2*y**6 + 15*r**2*s2**2*y**4*z - 24*r**2*s2**2*y**4 + 18*r**2*s2**2*y**2*z - 32*r**2*s2**2*y**2 - 5*r**2*s2**2*z + 8*r**2*s2**2 - 184*r**2*y**6*z + 128*r**2*y**6 + 114*r**2*y**4*z - 96*r**2*y**4 + 28*r**2*y**2*z - 6*r**2*z + 4*s2**2*y**8*z**2 + 12*s2**2*y**8*z - 16*s2**2*y**8 - 3*s2**2*y**6*z**2 - 6*s2**2*y**6*z + 8*s2**2*y**6 - 2*s2**2*y**4*z**2 - 12*s2**2*y**4*z + 16*s2**2*y**4 + s2**2*y**2*z**2 + 6*s2**2*y**2*z - 8*s2**2*y**2 - 8*y**8*z**2 + 40*y**8*z - 32*y**8 + 6*y**6*z**2 - 36*y**6*z + 32*y**6 + 4*y**4*z**2 - 8*y**4*z - 2*y**2*z**2 + 4*y**2*z
        x2 = 144*r**9*s2*y**2*zw - 72*r**9*s2*zw + 24*r**8*s*s2**2*y**3 - 48*r**8*s*s2**2*y + 432*r**8*s*y**3 - 432*r**8*s*y + 192*r**7*s2*y**4*z*zw + 192*r**7*s2*y**4*zw - 576*r**7*s2*y**2*z*zw - 576*r**7*s2*y**2*zw + 160*r**7*s2*z*zw + 160*r**7*s2*zw + 6*r**6*s*s2**2*y**5*z - 80*r**6*s*s2**2*y**3*z + 50*r**6*s*s2**2*y*z + 420*r**6*s*y**5*z - 1632*r**6*s*y**3*z + 620*r**6*s*y*z + 12*r**5*s2*y**6*z**2*zw - 12*r**5*s2*y**6*zw - 276*r**5*s2*y**4*z**2*zw + 276*r**5*s2*y**4*zw + 280*r**5*s2*y**2*z**2*zw - 280*r**5*s2*y**2*zw - 56*r**5*s2*z**2*zw + 56*r**5*s2*zw - 18*r**4*s*s2**2*y**7*z**2 + 14*r**4*s*s2**2*y**5*z**2 + 40*r**4*s*s2**2*y**5*z - 12*r**4*s*s2**2*y**3*z**2 - 60*r**4*s*s2**2*y**3*z + 12*r**4*s*s2**2*y*z**2 + 20*r**4*s*s2**2*y*z - 12*r**4*s*y**7*z**2 - 48*r**4*s*y**7*z - 660*r**4*s*y**5*z**2 + 1344*r**4*s*y**5*z + 744*r**4*s*y**3*z**2 - 1480*r**4*s*y**3*z - 160*r**4*s*y*z**2 + 344*r**4*s*y*z - 32*r**3*s2*y**6*z**3*zw + 96*r**3*s2*y**6*z**2*zw + 32*r**3*s2*y**6*z*zw - 96*r**3*s2*y**6*zw + 60*r**3*s2*y**4*z**3*zw - 180*r**3*s2*y**4*z**2*zw - 60*r**3*s2*y**4*z*zw + 180*r**3*s2*y**4*zw - 32*r**3*s2*y**2*z**3*zw + 96*r**3*s2*y**2*z**2*zw + 32*r**3*s2*y**2*z*zw - 96*r**3*s2*y**2*zw + 4*r**3*s2*z**3*zw - 12*r**3*s2*z**2*zw - 4*r**3*s2*z*zw + 12*r**3*s2*zw + 20*r**2*s*s2**2*y**7*z**3 - 74*r**2*s*s2**2*y**7*z**2 - 8*r**2*s*s2**2*y**7*z - 40*r**2*s*s2**2*y**5*z**3 + 145*r**2*s*s2**2*y**5*z**2 + 20*r**2*s*s2**2*y**5*z + 24*r**2*s*s2**2*y**3*z**3 - 84*r**2*s*s2**2*y**3*z**2 - 16*r**2*s*s2**2*y**3*z - 4*r**2*s*s2**2*y*z**3 + 13*r**2*s*s2**2*y*z**2 + 4*r**2*s*s2**2*y*z - 40*r**2*s*y**7*z**3 + 420*r**2*s*y**7*z**2 - 560*r**2*s*y**7*z + 80*r**2*s*y**5*z**3 - 810*r**2*s*y**5*z**2 + 1080*r**2*s*y**5*z - 48*r**2*s*y**3*z**3 + 456*r**2*s*y**3*z**2 - 608*r**2*s*y**3*z + 8*r**2*s*y*z**3 - 66*r**2*s*y*z**2 + 88*r**2*s*y*z - r*s2*y**8*z**4*zw + 8*r*s2*y**8*z**3*zw - 14*r*s2*y**8*z**2*zw - 8*r*s2*y**8*z*zw + 15*r*s2*y**8*zw + 3*r*s2*y**6*z**4*zw - 24*r*s2*y**6*z**3*zw + 42*r*s2*y**6*z**2*zw + 24*r*s2*y**6*z*zw - 45*r*s2*y**6*zw - 3*r*s2*y**4*z**4*zw + 24*r*s2*y**4*z**3*zw - 42*r*s2*y**4*z**2*zw - 24*r*s2*y**4*z*zw + 45*r*s2*y**4*zw + r*s2*y**2*z**4*zw - 8*r*s2*y**2*z**3*zw + 14*r*s2*y**2*z**2*zw + 8*r*s2*y**2*z*zw - 15*r*s2*y**2*zw - 10*s*s2**2*y**9*z**3 + 24*s*s2**2*y**9*z**2 + 30*s*s2**2*y**7*z**3 - 72*s*s2**2*y**7*z**2 - 30*s*s2**2*y**5*z**3 + 72*s*s2**2*y**5*z**2 + 10*s*s2**2*y**3*z**3 - 24*s*s2**2*y**3*z**2 + 20*s*y**9*z**3 - 88*s*y**9*z**2 + 96*s*y**9*z - 60*s*y**7*z**3 + 264*s*y**7*z**2 - 288*s*y**7*z + 60*s*y**5*z**3 - 264*s*y**5*z**2 + 288*s*y**5*z - 20*s*y**3*z**3 + 88*s*y**3*z**2 - 96*s*y**3*z
        x3 = -16*r**4*s2**2*y**4 - 112*r**4*s2**2*y**2 + 40*r**4*s2**2 - 48*r**4*y**4 - 352*r**4*y**2 + 64*r**4 - 44*r**2*s2**2*y**6*z + 80*r**2*s2**2*y**6 - 34*r**2*s2**2*y**4*z + 64*r**2*s2**2*y**4 + 80*r**2*s2**2*y**2*z - 144*r**2*s2**2*y**2 - 18*r**2*s2**2*z + 32*r**2*s2**2 - 232*r**2*y**6*z + 160*r**2*y**6 - 60*r**2*y**4*z + 224*r**2*y**2*z - 96*r**2*y**2 - 28*r**2*z + 4*s2**2*y**8*z**2 + 12*s2**2*y**8*z - 16*s2**2*y**8 - 2*s2**2*y**6*z**2 - 5*s2**2*y**4*z**2 - 30*s2**2*y**4*z + 40*s2**2*y**4 + 4*s2**2*y**2*z**2 + 24*s2**2*y**2*z - 32*s2**2*y**2 - s2**2*z**2 - 6*s2**2*z + 8*s2**2 - 8*y**8*z**2 + 40*y**8*z - 32*y**8 + 4*y**6*z**2 - 32*y**6*z + 32*y**6 + 10*y**4*z**2 - 20*y**4*z - 8*y**2*z**2 + 16*y**2*z + 2*z**2 - 4*z
        x4 = 24*r**9*s2*zw + 24*r**8*s*s2**2*y + 192*r**8*s*y + 384*r**7*s2*y**2*z*zw + 384*r**7*s2*y**2*zw - 96*r**7*s2*z*zw - 96*r**7*s2*zw + 84*r**6*s*s2**2*y**3*z - 60*r**6*s*s2**2*y*z + 1272*r**6*s*y**3*z - 552*r**6*s*y*z + 408*r**5*s2*y**4*z**2*zw - 408*r**5*s2*y**4*zw - 552*r**5*s2*y**2*z**2*zw + 552*r**5*s2*y**2*zw + 160*r**5*s2*z**2*zw - 160*r**5*s2*zw + 56*r**4*s*s2**2*y**5*z**2 - 48*r**4*s*s2**2*y**5*z + 60*r**4*s*s2**2*y**3*z**2 + 88*r**4*s*s2**2*y**3*z - 76*r**4*s*s2**2*y*z**2 - 40*r**4*s*s2**2*y*z + 1080*r**4*s*y**5*z**2 - 1920*r**4*s*y**5*z - 1320*r**4*s*y**3*z**2 + 2736*r**4*s*y**3*z + 352*r**4*s*y*z**2 - 880*r**4*s*y*z + 40*r**3*s2*y**6*z**3*zw - 120*r**3*s2*y**6*z**2*zw - 40*r**3*s2*y**6*z*zw + 120*r**3*s2*y**6*zw - 128*r**3*s2*y**4*z**3*zw + 384*r**3*s2*y**4*z**2*zw + 128*r**3*s2*y**4*z*zw - 384*r**3*s2*y**4*zw + 120*r**3*s2*y**2*z**3*zw - 360*r**3*s2*y**2*z**2*zw - 120*r**3*s2*y**2*z*zw + 360*r**3*s2*y**2*zw - 32*r**3*s2*z**3*zw + 96*r**3*s2*z**2*zw + 32*r**3*s2*z*zw - 96*r**3*s2*zw - 24*r**2*s*s2**2*y**7*z**3 + 90*r**2*s*s2**2*y**7*z**2 + 8*r**2*s*s2**2*y**7*z + 80*r**2*s*s2**2*y**5*z**3 - 296*r**2*s*s2**2*y**5*z**2 - 32*r**2*s*s2**2*y**5*z - 80*r**2*s*s2**2*y**3*z**3 + 290*r**2*s*s2**2*y**3*z**2 + 40*r**2*s*s2**2*y**3*z + 24*r**2*s*s2**2*y*z**3 - 84*r**2*s*s2**2*y*z**2 - 16*r**2*s*s2**2*y*z + 48*r**2*s*y**7*z**3 - 516*r**2*s*y**7*z**2 + 688*r**2*s*y**7*z - 160*r**2*s*y**5*z**3 + 1680*r**2*s*y**5*z**2 - 2240*r**2*s*y**5*z + 160*r**2*s*y**3*z**3 - 1620*r**2*s*y**3*z**2 + 2160*r**2*s*y**3*z - 48*r**2*s*y*z**3 + 456*r**2*s*y*z**2 - 608*r**2*s*y*z + r*s2*y**8*z**4*zw - 8*r*s2*y**8*z**3*zw + 14*r*s2*y**8*z**2*zw + 8*r*s2*y**8*z*zw - 15*r*s2*y**8*zw - 6*r*s2*y**6*z**4*zw + 48*r*s2*y**6*z**3*zw - 84*r*s2*y**6*z**2*zw - 48*r*s2*y**6*z*zw + 90*r*s2*y**6*zw + 10*r*s2*y**4*z**4*zw - 80*r*s2*y**4*z**3*zw + 140*r*s2*y**4*z**2*zw + 80*r*s2*y**4*z*zw - 150*r*s2*y**4*zw - 6*r*s2*y**2*z**4*zw + 48*r*s2*y**2*z**3*zw - 84*r*s2*y**2*z**2*zw - 48*r*s2*y**2*z*zw + 90*r*s2*y**2*zw + r*s2*z**4*zw - 8*r*s2*z**3*zw + 14*r*s2*z**2*zw + 8*r*s2*z*zw - 15*r*s2*zw + 10*s*s2**2*y**9*z**3 - 24*s*s2**2*y**9*z**2 - 60*s*s2**2*y**7*z**3 + 144*s*s2**2*y**7*z**2 + 100*s*s2**2*y**5*z**3 - 240*s*s2**2*y**5*z**2 - 60*s*s2**2*y**3*z**3 + 144*s*s2**2*y**3*z**2 + 10*s*s2**2*y*z**3 - 24*s*s2**2*y*z**2 - 20*s*y**9*z**3 + 88*s*y**9*z**2 - 96*s*y**9*z + 120*s*y**7*z**3 - 528*s*y**7*z**2 + 576*s*y**7*z - 200*s*y**5*z**3 + 880*s*y**5*z**2 - 960*s*y**5*z + 120*s*y**3*z**3 - 528*s*y**3*z**2 + 576*s*y**3*z - 20*s*y*z**3 + 88*s*y*z**2 - 96*s*y*z
        x5 = 48*r**4*s2**2*y**2 - 12*r**4*s2**2 + 144*r**4*y**2 - 24*r**4 + 65*r**2*s2**2*y**4*z - 120*r**2*s2**2*y**4 - 70*r**2*s2**2*y**2*z + 128*r**2*s2**2*y**2 + 13*r**2*s2**2*z - 24*r**2*s2**2 + 270*r**2*y**4*z - 160*r**2*y**4 - 244*r**2*y**2*z + 128*r**2*y**2 + 22*r**2*z - s2**2*y**6*z**2 - 6*s2**2*y**6*z + 8*s2**2*y**6 + 8*s2**2*y**4*z**2 + 36*s2**2*y**4*z - 48*s2**2*y**4 - 9*s2**2*y**2*z**2 - 42*s2**2*y**2*z + 56*s2**2*y**2 + 2*s2**2*z**2 + 12*s2**2*z - 16*s2**2 + 2*y**6*z**2 - 4*y**6*z - 16*y**4*z**2 + 56*y**4*z - 32*y**4 + 18*y**2*z**2 - 60*y**2*z + 32*y**2 - 4*z**2 + 8*z
        x6 = 12*r**5*s2*y**2*z**2*zw - 12*r**5*s2*y**2*zw - 4*r**5*s2*z**2*zw + 4*r**5*s2*zw - 34*r**4*s*s2**2*y**3*z**2 - 4*r**4*s*s2**2*y**3*z + 22*r**4*s*s2**2*y*z**2 + 4*r**4*s*s2**2*y*z - 12*r**4*s*y**3*z**2 - 72*r**4*s*y**3*z + 4*r**4*s*y*z**2 + 40*r**4*s*y*z + 20*r**3*s2*y**4*z**3*zw - 60*r**3*s2*y**4*z**2*zw - 20*r**3*s2*y**4*z*zw + 60*r**3*s2*y**4*zw - 32*r**3*s2*y**2*z**3*zw + 96*r**3*s2*y**2*z**2*zw + 32*r**3*s2*y**2*z*zw - 96*r**3*s2*y**2*zw + 12*r**3*s2*z**3*zw - 36*r**3*s2*z**2*zw - 12*r**3*s2*z*zw + 36*r**3*s2*zw - 12*r**2*s*s2**2*y**5*z**3 + 45*r**2*s*s2**2*y**5*z**2 + 4*r**2*s*s2**2*y**5*z + 20*r**2*s*s2**2*y**3*z**3 - 74*r**2*s*s2**2*y**3*z**2 - 8*r**2*s*s2**2*y**3*z - 8*r**2*s*s2**2*y*z**3 + 29*r**2*s*s2**2*y*z**2 + 4*r**2*s*s2**2*y*z + 24*r**2*s*y**5*z**3 - 258*r**2*s*y**5*z**2 + 344*r**2*s*y**5*z - 40*r**2*s*y**3*z**3 + 420*r**2*s*y**3*z**2 - 560*r**2*s*y**3*z + 16*r**2*s*y*z**3 - 162*r**2*s*y*z**2 + 216*r**2*s*y*z + r*s2*y**6*z**4*zw - 8*r*s2*y**6*z**3*zw + 14*r*s2*y**6*z**2*zw + 8*r*s2*y**6*z*zw - 15*r*s2*y**6*zw - 3*r*s2*y**4*z**4*zw + 24*r*s2*y**4*z**3*zw - 42*r*s2*y**4*z**2*zw - 24*r*s2*y**4*z*zw + 45*r*s2*y**4*zw + 3*r*s2*y**2*z**4*zw - 24*r*s2*y**2*z**3*zw + 42*r*s2*y**2*z**2*zw + 24*r*s2*y**2*z*zw - 45*r*s2*y**2*zw - r*s2*z**4*zw + 8*r*s2*z**3*zw - 14*r*s2*z**2*zw - 8*r*s2*z*zw + 15*r*s2*zw + 10*s*s2**2*y**7*z**3 - 24*s*s2**2*y**7*z**2 - 30*s*s2**2*y**5*z**3 + 72*s*s2**2*y**5*z**2 + 30*s*s2**2*y**3*z**3 - 72*s*s2**2*y**3*z**2 - 10*s*s2**2*y*z**3 + 24*s*s2**2*y*z**2 - 20*s*y**7*z**3 + 88*s*y**7*z**2 - 96*s*y**7*z + 60*s*y**5*z**3 - 264*s*y**5*z**2 + 288*s*y**5*z - 60*s*y**3*z**3 + 264*s*y**3*z**2 - 288*s*y**3*z + 20*s*y*z**3 - 88*s*y*z**2 + 96*s*y*z
        x7 = -5*s2**2*y**4*z**2 - 18*s2**2*y**4*z + 24*s2**2*y**4 + 6*s2**2*y**2*z**2 + 24*s2**2*y**2*z - 32*s2**2*y**2 - s2**2*z**2 - 6*s2**2*z + 8*s2**2 + 10*y**4*z**2 - 44*y**4*z + 32*y**4 - 12*y**2*z**2 + 48*y**2*z - 32*y**2 + 2*z**2 - 4*z
        x8 = r*s2*y**4*z**4*zw - 8*r*s2*y**4*z**3*zw + 14*r*s2*y**4*z**2*zw + 8*r*s2*y**4*z*zw - 15*r*s2*y**4*zw - 2*r*s2*y**2*z**4*zw + 16*r*s2*y**2*z**3*zw - 28*r*s2*y**2*z**2*zw - 16*r*s2*y**2*z*zw + 30*r*s2*y**2*zw + r*s2*z**4*zw - 8*r*s2*z**3*zw + 14*r*s2*z**2*zw + 8*r*s2*z*zw - 15*r*s2*zw + 10*s*s2**2*y**5*z**3 - 24*s*s2**2*y**5*z**2 - 20*s*s2**2*y**3*z**3 + 48*s*s2**2*y**3*z**2 + 10*s*s2**2*y*z**3 - 24*s*s2**2*y*z**2 - 20*s*y**5*z**3 + 88*s*y**5*z**2 - 96*s*y**5*z + 40*s*y**3*z**3 - 176*s*y**3*z**2 + 192*s*y**3*z - 20*s*y*z**3 + 88*s*y*z**2 - 96*s*y*z
        y1 = -5*r**4*s2**2 - 50*r**4 + 4*r**2*s2**2*z + 40*r**2*z + s2**2*z - 5*z**2 + 10*z
        y2 = -18*r**6 + 40*r**4*z + 40*r**4 - 14*r**2*z**2 + 14*r**2 + z**3 - 3*z**2 - z + 3
        y3 = -8*r**6*s2**2 - 192*r**6 + 30*r**4*s2**2*z + 500*r**4*z + 12*r**2*s2**2*z**2 + 20*r**2*s2**2*z - 160*r**2*z**2 + 344*r**2*z - 4*s2**2*z**3 + 13*s2**2*z**2 + 4*s2**2*z + 8*z**3 - 66*z**2 + 88*z
        y4 = 24*r**8 - 96*r**6*z - 96*r**6 + 160*r**4*z**2 - 160*r**4 - 32*r**2*z**3 + 96*r**2*z**2 + 32*r**2*z - 96*r**2 + z**4 - 8*z**3 + 14*z**2 + 8*z - 15
        y5 = 24*r**8 - 2*r**6*s2**2*z - 108*r**6*z - 22*r**4*s2**2*z**2 - 16*r**4*s2**2*z + 176*r**4*z**2 - 416*r**4*z + 12*r**2*s2**2*z**3 - 42*r**2*s2**2*z**2 - 8*r**2*s2**2*z - 24*r**2*z**3 + 228*r**2*z**2 - 304*r**2*z + 5*s2**2*z**3 - 12*s2**2*z**2 - 10*z**3 + 44*z**2 - 48*z
        y6 = -4*r**4*z**2 + 4*r**4 + 12*r**2*z**3 - 36*r**2*z**2 - 12*r**2*z + 36*r**2 - z**4 + 8*z**3 - 14*z**2 - 8*z + 15
        y7 = 6*r**4*s2**2*z + 4*r**4*z + 16*r**4 - 8*r**2*s2**2*z**2 + 29*r**2*s2**2*z + 4*r**2*s2**2 + 16*r**2*z**2 - 162*r**2*z + 216*r**2 - 10*s2**2*z**2 + 24*s2**2*z + 20*z**2 - 88*z + 96
        y8 = z**4 - 8*z**3 + 14*z**2 + 8*z - 15
        y9 = 5*s2**2*z**2 - 12*s2**2*z - 10*z**2 + 44*z - 48
        valx = x**8*x8 + 2*x**6*x6 + x**4*x4 + 2*x**2*x2 + 2*c*r**2*x**7*z*x7 + 2*c*r**2*x**5*z*x5 + 2*c*r**2*x**3*z*x3 - 2*c*r**2*x*z*x1 
        valy = - 8*r**4*s*y*y1 + 8*r**3*s2*y**2*zw*y2 + 2*r**2*s*y**3*y3 + r*s2*y**8*zw*y8 + 2*r*s2*y**6*zw*y6 + r*s2*y**4*zw*y4 + 2*s*y**9*z*y9 + 2*s*y**7*z*y7 + 2*s*y**5*y5 

        val =  r**(z - 8)*s2*zw1*(valx + valy + 80*r**9*s2*zw - 64*r**7*s2*z*zw - 64*r**7*s2*zw + 8*r**5*s2*z**2*zw - 8*r**5*s2*zw)/((z - 1)*(z + 1))

        return val
