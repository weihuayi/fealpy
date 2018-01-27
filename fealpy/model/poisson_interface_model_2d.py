import numpy as np

class CircleInterfaceData:
    def __init__(self, c, r, b1, b2):
        from ..mesh.implicit_curve import Circle  
        self.b1 = b1
        self.b2 = b2
        self.interface = Circle(radius=0.5) 

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
        val[flag[0]] = np.sin(pi*x)*np.sin(pi*y)-1

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[flag[1]] = np.cos(pi*x)*np.cos(pi*y)-1
        return val
    
    def gradient(self,p):
        pi = np.pi
        flag = self.subdomain(p)
        val = np.zeros(p.shape, dtype = np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        val[..., flag[0], 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., flag[0], 1] = pi*np.sin(pi*x)*np.cos(pi*y) 

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[..., flag[1], 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., flag[1], 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val 

    def source(self, p):
        pi = np.pi
        b1 = self.b1
        b2 = self.b2
        flag = self.subdomain(p)
        val = np.zeros(p.shape[0:-1], dtype=np.float)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        val[flag[0]] = 2*pi**2*b1*np.sin(pi*x)*np.sin(pi*y) 

        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        val[flag[1]] = 2*pi**2*b2*np.cos(pi*x)*np.cos(pi*y) 
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def func_jump(self, p):
        d, n = self.interface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        val0 = np.sin(pi*x)*np.sin(pi*y)
        val1 = np.cos(pi*x)*np.cos(pi*y)
        return val0 - val1

    def flux_jump(self, p):
        d, n = self.interface.project(p)
        x = p[..., 0]
        y = p[..., 1]
        val0 = np.zeros(p.shape, dtype = np.float)
        val0[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val0[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y) 
        val1 = np.zeros(p.shape, dtype = np.float)
        val1[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val1[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)

        b1 = self.b1
        b2 = self.b2
        val = b1*np.sum(val0*n, axis=-1) - b2*np.sum(val1*n, axis=-1)
        return val 
        
