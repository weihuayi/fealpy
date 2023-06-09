from fealpy.functionspace import ConformingScalarVESpace2d

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace import ConformingScalarVESpace2d

class ConformingScalarVEMLaplaceIntegrator2d():
    def __init__(self, projector, c=None):
        self.coef = c
        self.projector = projector

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d):
        p = space.p
        mesh = space.mesh
        coef = self.coef

        PI1 = self.projector.assembly_cell_matrix(space)
        D = self.projector.D

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
            G = self.projector.G
            tG = list(map(f, G))
            if coef is None:
                f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(D, PI1, tG)))
            else:
                pass

        return K
       
