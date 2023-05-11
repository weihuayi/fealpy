import numpy as np

from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh

class InHomogeneousData:
    def __init__(self, delta=0.02):
        self.delta = delta

    def domain(self):
        return np.array([-1, 1, -1, 1])

    def init_mesh(self, n=4):
        """ generate the initial mesh
        """
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float64)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh


    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros_like(p)
        d=x**2+y**2+self.delta
        val[..., 0] = y/d
        val[..., 1] = (-x)/d
        return val 

    @cartesian
    def curl(self, p):
        """ The curl of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        val = -2*self.delta/(x**2+y**2+self.delta)**2
        return val 
    
    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        mu=1+x**2+y**2
        d=x**2+y**2+self.delta
        val = np.zeros_like(p)  
        val[..., 0] =(-2*self.delta)*(2*y*d**2-mu*2*d*2*y)/d**4- y/d
        val[..., 1] = 2*self.delta*(2*x*d**2-mu*2*d*2*x)/d**4 - (-x)/d
        return val
		
    @cartesian
    def dirichlet(self, p, t):
        """
        Notes
        -----
        p : (NQ, NE, GD)
        t:  (NE, GD)
        """
        val = np.sum(self.solution(p)*t, axis=-1)
        return val
	 

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        return np.abs(x) < 1e-12 

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def inv_mu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val=1+x**2+y**2
        return val

    @cartesian
    def epsilon(self, p):
        x = p[..., 0]
        y = p[..., 1]
        shape = p.shape + (2, )
        val = np.zeros(shape, dtype=p.dtype)
        val[..., 0, 0] = 1 + x**2
        val[..., 0, 1] = x*y
        val[..., 1, 0] = x*y
        val[..., 1, 1] = 1+y**2
        return val


class CosSinData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype == 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float64)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int_)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'squad':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)


    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros_like(p)
        val[..., 0] = np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = -np.sin(pi*x)*np.cos(pi*y)
        return val 

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros_like(p)  
        val[..., 0] = (2*pi**2 - 1)*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = (-2*pi**2+1)*np.sin(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def curl(self, p):
        """ The curl of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = -2*pi*np.cos(pi*x)*np.cos(pi*y)
        return val 

    @cartesian
    def dirichlet(self, p, t):
        """
        Notes
        -----
        p : (NQ, NE, GD)
        t:  (NE, GD)
        """
        val = np.sum(self.solution(p)*t, axis=-1)
        return val 

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        return np.abs(x) < 1e-12 

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val


class LShapeRSinData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float64)
        if meshtype == 'tri':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int_)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def domain(self):
        points = [[0, 0], [1, 0], [1, 1], [-1, 1], [-1, -1], [0, -1]]
        facets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        return points, facets

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float)
        """
        x = p[..., 0]
        y = p[..., 1]
        
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        beta=2/3
        val = np.zeros_like(p)
        
        # the true solution we need 
        val[..., 0] = beta*(x*x+y*y)**((beta-1)/2)*np.sin((beta-1)*theta)
        val[..., 1] = beta*(x*x+y*y)**((beta-1)/2)*np.cos((beta-1)*theta)
        
        return val 

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        
        
        pi = np.pi
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        beta=2/3
        val = np.zeros_like(p) 
        
        # the source we need
        val[..., 0] = -beta*(x*x+y*y)**((beta-1)/2)*np.sin((beta-1)*theta)
        val[..., 1] = -beta*(x*x+y*y)**((beta-1)/2)*np.cos((beta-1)*theta)
        
        return val

    @cartesian
    def curl(self, p):
        """ The curl of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        # the curl we need 
        val = 0
        return val 

    @cartesian
    def curlcurl(self, p):
        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        beta=2/3
        
        val = np.zeros_like(p)
        # the curl curl we need
        val[...,0] = 0
        val[...,1] = 0
        return val

    @cartesian
    def dirichlet(self, p, t):
        """
        Notes
        -----
        p : (NQ, NE, GD)
        t:  (NE, GD)
        """
        val = np.sum(self.solution(p)*t, axis=-1)
        return val 

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        return np.abs(x) < 1e-12 

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val
