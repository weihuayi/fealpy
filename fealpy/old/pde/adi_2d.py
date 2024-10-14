import numpy as np

from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh



class ADI_2d:
    def __init__(self,sigma,epsilon,mu):
        self.sigma=sigma
        self.epsilon=epsilon
        self.mu=mu

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


    
    def Efield(self, p, t):
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
        pi=np.pi
        val[..., 0] = np.exp(-pi*t)*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = -np.exp(-pi*t)*np.sin(pi*x)*np.cos(pi*y)
        return val 

   
    def Hz(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val =np.exp(-pi*t)*np.cos(pi*x)*np.cos(pi*y)
        return val

    
    def gsource(self, p, t):
        """ The curl of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sigma=self.sigma
        val = np.zeros_like(p)
        valE=self.Efield(p, t)
        val=sigma*valE
        return val 

    
    def fzsource(self, p, t):
        """
        Notes
        -----
        p : (NQ, NE, GD)
        t:  (NE, GD)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi=np.pi
        val = -3*pi*np.exp(-pi*t)*np.cos(pi*x)*np.cos(pi*y)
        return val
        
    @cartesian
    def init_H_value(self,p):
    	return self.Hz(p, 0.0)
    
    @cartesian
    def init_E_value(self, p):
        return self.Efield(p, 0.0)
    

    


