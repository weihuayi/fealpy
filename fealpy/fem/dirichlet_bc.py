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

    def apply(self, A, f, uh):
        """
        @brief 处理 Dirichlet 边界条件  

        @note 
            * 如果 `uh` 是一个向量场或张量场，则 `f` 必须是一个展平向量，F.shape[0] = A.shape[0] == A.shape[1]
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.apply_for_vspace_with_scalar_basis(A, f, uh)
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.apply_for_sspace_and_vspace_with_vector_basis(A, f, uh)

    def apply_for_sspace_and_vspace_with_vector_basis(self, A, f, uh):
        """
        """
        space = self.space
        gD = self.gD
        isDDof = space.boundary_interpolate(gD, uh, threshold=self.threshold) # isDDof.shape == uh.shape
        f = f - A@uh # 注意这里不修改外界 f 的值

        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1

        f[isDDof] = uh[isDDof]

        return A, f 

    def apply_for_vspace_with_scalar_basis(self, A, f, uh):
        """
        @brief 
        """
        space = self.space
        assert isinstance(space, tuple) and not isinstance(space[0], tuple)

        gD = self.gD
        isDDof = space[0].set_dirichlet_bc(gD, uh, threshold=self.threshold) # isDDof.shape == uh.shape
        f = f - A@uh.flat # 注意这里不修改外界 f 的值

        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof.flat] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        f[isDDof.flat] = uh[isDDof].flat
        return A, f 
