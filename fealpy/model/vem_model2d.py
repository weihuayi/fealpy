import numpy as np

from ..mesh.TriangleMesh import TriangleMesh  
from ..mesh.tree_data_structure import Quadtree 

class ffData:
    def __init__(self):
        pass

    def init_mesh(self, n=4, meshtype='quadtree'):
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

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        
        f = np.zeros(p.shape[0:-1])
        
        for i in range(x.shape[0]):         
            for j in range(5):
                for k in range(5):
                    if (np.mod(j+k, 2) == 1) and  np.any(np.bitwise_and((j)/4 < x[i], (j+1)/4 > x[i])) and np.any(np.bitwise_and((k-1)/4 < y[i], k/4 > y[i])):                
                        f[..., i] = 1
                        break 
            if np.any(f[..., i] == 0):
                f[..., i] = -1
                  
        return f

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        pass

  



