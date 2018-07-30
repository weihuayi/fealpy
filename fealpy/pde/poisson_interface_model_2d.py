import numpy as np


from ..mesh.TriangleMesh import TriangleMesh  
from ..mesh.tree_data_structure import Quadtree 

class CircleInterfaceDataTest:
    def __init__(self, c, r, b1, b2):
        from ..mesh.implicit_curve import Circle  
        self.b1 = b1
        self.b2 = b2
        self.interface = Circle(center=c, radius=r) 

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

    def diffusion_coefficient(self, p):
        flag = self.subdomain(p)
        k = np.zeros(p.shape[:-1], dtype=np.float)
        k[flag[0]] = self.b1  
        k[flag[1]] = self.b2
        k = np.ones(p.shape[:-1], dtype=np.float)
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        sdflag = [self.interface(p) > 0, self.interface(p) < 0]
        return sdflag 

    def solution_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return np.sin(pi*x)*np.sin(pi*y) - 1

    def solution_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return np.sin(pi*x)*np.sin(pi*y) - 1

    def solution(self, p):
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        val[flag[0]] = self.solution_plus(p[flag[0], :]) 
        val[flag[1]] = self.solution_minus(p[flag[1], :]) 

        val = self.solution_plus(p)
        return val

    def gradient_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y) 
        return val 

    def gradient_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y) 
        return val
    
    def gradient(self,p):
        pi = np.pi
        flag = self.subdomain(p)
        val = np.zeros(p.shape, dtype = np.float)
        val[..., flag[0], :] = self.gradient_plus(p[flag[0], :]) 
        val[..., flag[1], :] = self.gradient_minus(p[flag[1], :])
        return val 

    def source_plus(self, p):
        pi = np.pi
        b1 = self.b1
        return 2*pi**2*self.solution_plus(p)

    def source_minus(self, p):
        pi = np.pi
        b2 = self.b2
        return 2*pi**2*b2*self.solution_minus(p)

    def source(self, p):
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        val[flag[0]] = self.source_plus(p[flag[0], :]) 
        val[flag[1]] = self.source_minus(p[flag[1], :]) 

        val = self.source_plus(p)
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def func_jump(self, p):
        d, n = self.interface.project(p)
        val0 = self.solution_plus(p) 
        val1 = self.solution_minus(p)
        return val0 - val1

    def flux_jump(self, p):
        d, n = self.interface.project(p)
        val0 = self.gradient_plus(p)
        val1 = self.gradient_minus(p)

        b1 = self.b1
        b2 = self.b2
        val = b1*np.sum(val0*n, axis=-1) - b2*np.sum(val1*n, axis=-1)
        return val 

class CircleInterfaceData:
    def __init__(self, c, r, b1, b2):
        from ..mesh.implicit_curve import Circle  
        self.b1 = b1
        self.b2 = b2
        self.interface = Circle(center=c, radius=r) 

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
        flag = self.subdomain(p)
        k = np.zeros(p.shape[:-1], dtype=np.float)
        k[flag[0]] = self.b1  
        k[flag[1]] = self.b2
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        sdflag = [self.interface(p) > 0, self.interface(p) < 0]
        return sdflag 

    def solution_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return np.sin(pi*x)*np.sin(pi*y) - 1

    def solution_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return np.cos(pi*x)*np.cos(pi*y) - 1

    def solution(self, p):
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        val[flag[0]] = self.solution_plus(p[flag[0], :]) 
        val[flag[1]] = self.solution_minus(p[flag[1], :]) 
        return val

    def gradient_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y) 
        return val 

    def gradient_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val
    
    def gradient(self,p):
        pi = np.pi
        flag = self.subdomain(p)
        val = np.zeros(p.shape, dtype = np.float)
        val[..., flag[0], :] = self.gradient_plus(p[flag[0], :]) 
        val[..., flag[1], :] = self.gradient_minus(p[flag[1], :])
        return val 

    def source_plus(self, p):
        pi = np.pi
        b1 = self.b1
        x = p[..., 0]
        y = p[..., 1]
        return 2*pi**2*b1*np.sin(pi*x)*np.sin(pi*y)

    def source_minus(self, p):
        pi = np.pi
        b2 = self.b2
        x = p[..., 0]
        y = p[..., 1]
        return 2*pi**2*b2*np.cos(pi*x)*np.cos(pi*y)

    def source(self, p):
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        val[flag[0]] = self.source_plus(p[flag[0], :]) 
        val[flag[1]] = self.source_minus(p[flag[1], :]) 
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def func_jump(self, p):
        d, n = self.interface.project(p)
        val0 = self.solution_plus(p) 
        val1 = self.solution_minus(p)
        return val0 - val1

    def flux_jump(self, p):
        d, n = self.interface.project(p)
        val0 = self.gradient_plus(p)
        val1 = self.gradient_minus(p)

        b1 = self.b1
        b2 = self.b2
        val = b1*np.sum(val0*n, axis=-1) - b2*np.sum(val1*n, axis=-1)
        return val 

class FoldCurveInterfaceData:
    def __init__(self, a, b1, b2):
        from ..mesh.implicit_curve import FoldCurve  
        self.b1 = b1
        self.b2 = b2
        self.interface = FoldCurve(a=a) 

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
        flag = self.subdomain(p)
        k = np.zeros(p.shape[:-1], dtype=np.float)
        k[flag[0]] = self.b1  
        k[flag[1]] = self.b2
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        sdflag = [self.interface(p) > 0, self.interface(p) < 0]
        return sdflag 

    def solution_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return np.sin(pi*x)*np.sin(pi*y) - 1

    def solution_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        return np.cos(pi*x)*np.cos(pi*y) - 1

    def solution(self, p):
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        val[flag[0]] = self.solution_plus(p[flag[0], :]) 
        val[flag[1]] = self.solution_minus(p[flag[1], :]) 
        return val

    def gradient_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y) 
        return val 

    def gradient_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype = np.float)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val
    
    def gradient(self,p):
        pi = np.pi
        flag = self.subdomain(p)
        val = np.zeros(p.shape, dtype = np.float)
        val[..., flag[0], :] = self.gradient_plus(p[flag[0], :]) 
        val[..., flag[1], :] = self.gradient_minus(p[flag[1], :])
        return val 

    def source_plus(self, p):
        pi = np.pi
        b1 = self.b1
        x = p[..., 0]
        y = p[..., 1]
        return 2*pi**2*b1*np.sin(pi*x)*np.sin(pi*y)

    def source_minus(self, p):
        pi = np.pi
        b2 = self.b2
        x = p[..., 0]
        y = p[..., 1]
        return 2*pi**2*b2*np.cos(pi*x)*np.cos(pi*y)

    def source(self, p):
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        val[flag[0]] = self.source_plus(p[flag[0], :]) 
        val[flag[1]] = self.source_minus(p[flag[1], :]) 
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def func_jump(self, p):
        d, n = self.interface.project(p)
        val0 = self.solution_plus(p) 
        val1 = self.solution_minus(p)
        return val0 - val1

    def flux_jump(self, p):
        d, n = self.interface.project(p)
        val0 = self.gradient_plus(p)
        val1 = self.gradient_minus(p)

        b1 = self.b1
        b2 = self.b2
        val = b1*np.sum(val0*n, axis=-1) - b2*np.sum(val1*n, axis=-1)
        return val 
        
class Circle1InterfaceData:
    def __init__(self, c, r, b1, b2, b, C):
        from ..mesh.implicit_curve import Circle  
        self.b1 = b1
        self.b2 = b2
        self.b = b
        self.C = C
        self.interface = Circle(radius=2) 

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
        flag = self.subdomain(p)
        k = np.zeros(p.shape[:-1], dtype=np.float)
        k[flag[0]] = self.b1  
        k[flag[1]] = self.b2
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        sdflag = [self.interface(p) > 0, self.interface(p) < 0]
        return sdflag 

    def solution(self, p):
        pi = np.pi
        b = self.b
        C = self.C
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        r = np.sqrt(x**2 + y**2)
        val[flag[0]] = (1 - 9/(8*b))/4 - (r**4/2+r**2)/b + C*np.log(2*r)/b

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[flag[1]] = x**2 + y**2
        return val
    
    def gradient(self,p):
        pi = np.pi
        b = self.b
        C = self.C
        flag = self.subdomain(p)
        val = np.zeros(p.shape, dtype = np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        r = np.sqrt(x**2 + y**2)
        val[..., flag[0], 0] = (2*r**2 + 2 + C/r**2)*(x/b)
        val[..., flag[0], 1] = (2*r**2 + 2 + C/r**2)*(y/b)

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[..., flag[1], 0] = 2*x
        val[..., flag[1], 1] = 2*y
        return val 

    def source(self, p):
        pi = np.pi
        b1 = self.b1
        b2 = self.b2
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        val[flag[0]] = 8*(x**2 + y**2) + 4 

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[flag[1]] = 8*(x**2 + y**2) + 4
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def func_jump(self, p):
        d, n = self.interface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        val0 = (1 - 9/(8*b))/4 - (r**4/2+r**2)/b + C*np.log(2*r)/b
        val1 = x**2 + y**2
        return val0 - val1

    def flux_jump(self, p):
        d, n = self.interface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        val0 = np.zeros(p.shape, dtype = np.float)
        val0[..., 0] = (2*r**2 + 2 + C/r**2)*(x/b)
        val0[..., 1] = (2*r**2 + 2 + C/r**2)*(y/b)
        val1 = np.zeros(p.shape, dtype = np.float)
        val1[..., 0] = 2*x
        val1[..., 1] = 2*y

        b1 = self.b1
        b2 = self.b2
        val = b1*np.sum(val0*n, axis=-1) - b2*np.sum(val1*n, axis=-1)
        return val 
class SquareInterfaceData:
    def __init__(self, c, r, b1, b2):
        from ..mesh.implicit_curve import Square 
        self.b1 = b1
        self.b2 = b2
        self.interface = Square() 

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
        flag = self.subdomain(p)
        k = np.zeros(p.shape[:-1], dtype=np.float)
        k[flag[0]] = self.b1  
        k[flag[1]] = self.b2
        return k

    def subdomain(self, p):
        """
        get the subdomain flag of the subdomain including point p.
        """
        sdflag = [self.interface(p) > 0, self.interface(p) < 0]
        return sdflag 

    def solution(self, p):
        pi = np.pi
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        val[flag[0]] = np.cos(y)*np.exp(x)

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[flag[1]] = np.cos(x) + np.cos(y)
        return val
    
    def gradient(self,p):
        pi = np.pi
        flag = self.subdomain(p)
        val = np.zeros(p.shape, dtype = np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        val[..., flag[0], 0] = np.cos(y)*np.exp(x)
        val[..., flag[0], 1] = -np.sin(y)*np.exp(x)

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[..., flag[1], 0] = -np.sin(x)
        val[..., flag[1], 1] = -np.sin(y)
        return val 

    def source(self, p):
        pi = np.pi
        b1 = self.b1
        b2 = self.b2
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        val[flag[0]] = 0

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[flag[1]] = -b2*(np.cos(x) + np.cos(y))
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def func_jump(self, p):
        d, n = self.interface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        val0 = np.cos(y)*np.exp(x)
        val1 = np.cos(x) + np.cos(y)
        return val0 - val1

    def flux_jump(self, p):
        d, n = self.interface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        val0 = np.zeros(p.shape, dtype = np.float)
        val0[..., 0] = np.cos(y)*np.exp(x)
        val0[..., 1] = -np.sin(y)*np.exp(x)
        val1 = np.zeros(p.shape, dtype = np.float)
        val1[..., 0] = -np.sin(x)
        val1[..., 1] = -np.sin(y)

        b1 = self.b1
        b2 = self.b2
        val = b1*np.sum(val0*n, axis=-1) - b2*np.sum(val1*n, axis=-1)
        return val 
            
