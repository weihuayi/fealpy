import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace import ConformingScalarVESpace2d
from ..functionspace import NonConformingScalarVESpace2d

class ConformingScalarVEMLaplaceIntegrator2d():
    def __init__(self, PI1, G, c=None):
        self.coef = c
        self.PI1 = PI1
        self.G = G

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d, out=None):
        p = space.p
        mesh = space.mesh
        coef = self.coef

        def f(x):
            x[0, :] = 0
            return x

        if p == 1:
            tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            if coef is None:
                f1 = lambda x: x[0].T@tG@x[0]
                K = list(map(f1, zip(self.PI1)))
            elif isinstance(coef, float):
                f1 = lambda x: coef*(x[0].T@tG@x[0])
                K = list(map(f1, zip(self.PI1)))
            elif isinstance(coef, np.ndarray):
                f1 = lambda x: x[1]*(x[0].T@tG@x[0])
                K = list(map(f1, zip(self.PI1, coef)))

        else:
            tG = list(map(f, self.G))
            if coef is None:
                f1 = lambda x: x[0].T@x[1]@x[0]
                K = list(map(f1, zip(self.PI1, tG)))
            elif isinstance(coef, float):
                f1 = lambda x: coef*(x[0].T@x[1]@x[0])
                K = list(map(f1, zip(self.PI1, tG)))
            elif isinstance(coef, np.ndarray):
                f1 = lambda x: x[2]*(x[0].T@x[1]@x[0])
                K = list(map(f1, zip(self.PI1, tG, coef)))
        if out is None:
            return K
        else:
            for i in range(len(K)):
                out[i] += K[i]

       

class NonConformingScalarVEMLaplaceIntegrator2d():
    def __init__(self, PI1, G, c=None):
        self.coef = c
        self.PI1 = PI1
        self.G = G 

    def assembly_cell_matrix(self, space: NonConformingScalarVESpace2d):
        """
        """

        def f(x):
            x[0, :] = 0
            return x
        tG = list(map(f, self.G)) # 注意这里把 G 修改掉了
        f1 = lambda x: x[0].T@x[1]@x[0] 
        K = list(map(f1, zip(self.PI1, tG)))
        return K
