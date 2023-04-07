import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat

class DirichletBC():
    """
    """
    def __init__(self, space, gD, threshold=None):
        self.space = space
        self.gD = gD
        self.threshold = threshold
        self.bctype = 'Dirichlet'

    def apply(self, A, F, uh):
        """
        @brief 处理 Dirichlet 边界条件  

        @note 如是 `uh` 是一个向量场或张量场，则 `F` 必须是一个展平向量，F.shape[0] = A.shape[0] == A.shape[1]
        """
        space = self.space
        gD = self.gD
        isDDof = space.set_dirichlet_bc(gD, uh, threshold=self.threshold) # isDDof.shape == uh.shape
        F -= A@uh.flat
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        F[isDDof.flat] = uh[isDDof].flat
        return A, F 
