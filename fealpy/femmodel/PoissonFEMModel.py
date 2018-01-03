import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

class PoissonFEMModel:
    def __init__(self, model, V, dtype=np.float):
        self.V = V
        self.model = model  
        self.lqf = lqf
        self.rqf = rqf

    def get_left_matrix(self):
        return laplace(self.V, self.lqf, a=None)

    def get_right_vector(self):
        return source(self.pde.source, self.V, self.rqf)

    def get_neuman_vector(self):
        pass
