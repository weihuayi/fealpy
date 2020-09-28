import numpy as np

from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh

class CosSinData:
    """
    -\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """
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
            (0, 1)], dtype=np.float)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
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
                (0.5, 0.45)], dtype=np.float)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
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
