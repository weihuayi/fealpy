import numpy as np
from scipy.sparse import csr_matrix

class BilinearForm:
    """

    """
    def __init__(self, space, atype=None):
        """
        @brief 
        """
        self.space = space
        self.atype = atype # 矩阵组装的方式，None、fast、ref
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

        self._M = None # 需要组装的矩阵 

    def add_domain_integrator(self, I):
        """
        @brief 增加一个区域积分对象
        """
        self.dintegrators.append(I)


    def add_boundary_integrator(self, I):
        """
        @brief 增加一个边界积分对象
        """
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
        @brief 当空间发生改变时，重新组装矩阵
        """

    def assembly(self):
        """
        @brief 数值积分组装

        @note space 可能是以下的情形
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        space = self.space
        if not isinstance(space, tuple): 
            ldof = space.number_of_local_dofs()
            gdof = space.number_of_global_dofs()

            mesh = space.mesh
            NC = mesh.number_of_cells()
            CM = np.zeros((NC, ldof, ldof), dtype=space.ftype)
            for inte in self.dintegrators:
                inte.assembly_cell_matrix(space, out=CM)

            cell2dof = space.cell_to_dof()
            I = np.broadcast_to(cell2dof[:, :, None], shape=CM.shape)
            J = np.broadcast_to(cell2dof[:, None, :], shape=CM.shape)

            self._M = csr_matrix((CM.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        else: 
            mesh = space[0].mesh
            GD = mesh.geo_dimension()
            NC = mesh.number_of_cells()
            ldof = space[0].number_of_local_dofs()
            gdof = space[0].number_of_global_dofs()
            cell2dof = space[0].cell_to_dof() # 标量

            CM = np.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)

            for inte in self.dintegrators:
                inte.assembly_cell_matrix(space, out=CM)

            self._M = csr_matrix((GD*gdof, GD*gdof), dtype=space[0].ftype)
            if space[0].doforder == 'sdofs':
                for i in range(GD):
                    for j in range(i, GD):
                        if i == j:
                            val = CM[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof]
                            I = np.broadcast_to(cell2dof[:, :, None]+i*gdof, shape=val.shape)
                            J = np.broadcast_to(cell2dof[:, None, :]+i*gdof, shape=val.shape)
                            self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
                        else:
                            val = CM[:, i*ldof:(i+1)*ldof, j*ldof:(j+1)*ldof]
                            I = np.broadcast_to(cell2dof[:, :, None]+i*gdof, shape=val.shape)
                            J = np.broadcast_to(cell2dof[:, None, :]+j*gdof, shape=val.shape)
                            self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))

                            val = CM[:, j*ldof:(j+1)*ldof, i*ldof:(i+1)*ldof]
                            self._M += csr_matrix((val.flat, (J.flat, I.flat)), shape=(GD*gdof, GD*gdof))

            elif space[0].doforder == 'vdims':
                for i in range(GD):
                    for j in range(i, GD):
                        if i==j:
                            val = CM[:, i::GD, i::GD]
                            I = np.broadcast_to(GD*cell2dof[:, :, None]+i, shape=val.shape)
                            J = np.broadcast_to(GD*cell2dof[:, None, :]+i, shape=val.shape)
                            self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
                        else:
                            val = CM[:, i::GD, j::GD] 
                            I = np.broadcast_to(GD*cell2dof[:, :, None]+i, shape=val.shape)
                            J = np.broadcast_to(GD*cell2dof[:, None, :]+j, shape=val.shape)
                            self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))

                            val = CM[:, j::GD, i::GD] 
                            self._M += csr_matrix((val.flat, (J.flat, I.flat)), shape=(GD*gdof, GD*gdof))



    def fast_assembly(self):
        """
        @brief 免数值积分组装
        """

    def parallel_assembly(self):
        """
        @brief 多线程数值积分组装
        @note 特别当三维情形，最好并行来组装
        """


