from fealpy.functionspace import ConformingScalarVESpace2d

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace import ConformingScalarVESpace2d

class ConformingScalarVEMLaplaceIntegrator():
    def __init__(self, G, D, PI1, c=None):
        self.coef = c
        self.G = G
        self.D = D
        self.PI1 = PI1

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d):
        p = space.p
        mesh = space.mesh
        coef = self.coef
        G = self.G
        D = self.D
        PI1 = self.PI1

        def f(x):
            x[0, :] = 0
            return x

        if p == 1:
            tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            if coef is None:
                f1 = lambda x: x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(D, PI1)))
            else:
                pass
            
        else:
            tG = list(map(f, G))
            if coef is None:
                f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(D, PI1, tG)))
            else:
                pass

        return K
       
