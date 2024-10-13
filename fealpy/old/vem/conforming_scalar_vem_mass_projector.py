
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace import ConformingScalarVESpace2d
from ..functionspace import NonConformingScalarVESpace2d

class ConformingScalarVEMMassIntegrator2d():
    def __init__(self, PI0, H, c=None):
        self.coef = 1 if c is None else c
        self.PI0 = PI0
        self.H = H

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d, out=None):
        f1 = lambda x: self.coef*x[0].T@x[1]@x[0] 
        K = list(map(f1, zip(self.PI0, self.H)))
        if out is None:
            return K
        else:
            for i in range(len(K)):
                out[i] += K[i]


