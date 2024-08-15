import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace import ConformingScalarVESpace2d
from ..functionspace import NonConformingScalarVESpace2d

class DofDofStabilizationTermIntegrator2d():
    def __init__(self, PI1, D):
        self.PI1 = PI1
        self.D = D

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d, out=None):
        p = space.p
        mesh = space.mesh

        f1 = lambda x: (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
        #def f1(x):
        #    print(np.max(np.abs(x[0])))
        #    print(np.max(np.abs(x[1])))
        #    print(x[0]@x[1])
        #    print((np.eye(x[1].shape[1]) - x[0]@x[1]).T)
        #    return (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])

        K = list(map(f1, zip(self.D, self.PI1)))

        if out is None:
            return K
        else:
            for i in range(len(K)):
                out[i] += K[i]

       
