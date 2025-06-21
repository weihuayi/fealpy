import numpy as np
from scipy.sparse import csr_matrix

class BilinearForm:
    def __init__(self, space, atype=None):
        """
        @brief 
        """
        self.space = space
        self.atype = atype # 矩阵组装的方式，None、fast、ref
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

        self._M = None # 需要组装的矩阵 
        self._K = None

    def add_domain_integrator(self, I) -> None:
        """
        @brief 增加一个或多个区域积分对象
        """
        if isinstance(I, list):
            self.dintegrators.extend(I)
        else:
            self.dintegrators.append(I)


    def add_boundary_integrator(self, I):
        """
        @brief 增加一个或多个边界积分对象
        """
        if isinstance(I, list):
            self.bintegrators.extend(I)
        else:
            self.bintegrators.append(I)

    def mult(self, x, out=None):
        """
        """
        if out is None:
            return self._M@x
        else:
            out[:] = self._M@x

    def add_mult(self, x, y, a=1.0):
        y += a*(self._M@x)

    def get_matrix(self, copy=False):
        if copy is False:
            return self._M
        else:
            return self._M.copy()

    def update(self):
        """
        @brief 当空间发生改变时，调用这个函数重新组装矩阵
        """
        self.assembly()

    def assembly(self):
        """
        @brief 数值积分组装

        """
        space = self.space
        K = self.dintegrators[0].assembly_cell_matrix(space)
        for i in range(len(self.dintegrators))[1:]:
            self.dintegrators[i].assembly_cell_matrix(space, out=K)
        self._K = K
        return self._assembly(K)

    def _assembly(self, K):
        space = self.space
        cell2dof = space.cell_to_dof() 
        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flat

        I = np.concatenate(list(map(f2, cell2dof)))
        J = np.concatenate(list(map(f3, cell2dof)))
        val = np.concatenate(list(map(f4, K)))
        gdof = space.number_of_global_dofs()
        self._M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float64)
        return self._M

