import numpy as np
from scipy.sparse import csr_matrix

class MixedBilinearForm:
    def __init__(self, trialspace:tuple, testspace:tuple, atype=None):
        """
        @brief 
        """
        self.trial_space = trialspace
        self.test_space = testspace
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
        @brief 当空间发生改变时，调用这个函数重新组装矩阵
        """
        self.assembly()

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
        if isinstance(self.trial_space, tuple) and not isinstance(self.trial_space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis()


    def assembly_for_sspace_and_vspace_with_vector_basis(self):
        """
        @brief 基函数为标量函数的标量空间, 以及基函数为向量函数的函数空间
        """
        space = self.space
        ldof = space.number_of_local_dofs()
        gdof = space.number_of_global_dofs()

        mesh = space.mesh
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix(space, out=CM)

        cell2dof = space.cell_to_dof()
        I = np.broadcast_to(cell2dof[:, :, None], shape=CM.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=CM.shape)
        self._M = csr_matrix((CM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

    def assembly_for_vspace_with_scalar_basis(self):
        """
        @brief 基函数由标量函数组合而成的向量函数空间
        """
        trial_space = self.trial_space[0]
        test_space = self.test_space[0]
        assert isinstance(self.trial_space, tuple) and not isinstance(self.trial_space[0], tuple)
        assert isinstance(self.test_space, tuple) and not isinstance(self.test_space[0], tuple)
        
        trial_D = len(self.trial_space)
        test_D = len(self.test_space)
        mesh = trial_space.mesh
    
        trial_ldof = trial_space.number_of_local_dofs()
        test_ldof = test_space.number_of_local_dofs()
        trial_gdof = trial_space.number_of_global_dofs()
        test_gdof = test_space.number_of_global_dofs()
        trial_cell2dof = trial_space.cell_to_dof() 
        test_cell2dof = test_space.cell_to_dof() 

        cellmeasure = mesh.entity_measure()
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, test_D*test_ldof, trial_D*trial_ldof), dtype=trial_space.ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix(self.trial_space, self.test_space, cellmeasure=cellmeasure, out=CM)

        self._M = csr_matrix((test_D*test_gdof,trial_D*trial_gdof), dtype=trial_space.ftype)
        
        if trial_space.doforder == 'sdofs': # 标量自由度排序优先
            for i in range(test_D):
                for j in range(trial_D):
                        val = CM[:, i*test_ldof:(i+1)*test_ldof, j*trial_ldof:(j+1)*trial_ldof]
                        I = np.broadcast_to(test_cell2dof[:, :, None]+i*test_gdof, shape=val.shape)
                        J = np.broadcast_to(trial_cell2dof[:, None, :]+j*trial_gdof, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(test_D*test_gdof,trial_D*trial_gdof))
        
        elif trial_space.doforder == 'vdims': # 向量分量自由度排序优先
            for i in range(test_D):
                for j in range(trial_D):
                    val = CM[:, i::test_D, j::trial_D]
                    I = np.broadcast_to(test_D*test_cell2dof[:, :, None] + i, shape=val.shape)
                    J = np.broadcast_to(trial_D*trial_cell2dof[:, None, :] + j, shape=val.shape)
                    self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(test_D*test_gdof, trial_D*trial_gdof))
        
        return self._M


    def fast_assembly(self):
        """
        @brief 免数值积分组装
        """

    def parallel_assembly(self):
        """
        @brief 多线程数值积分组装
        @note 特别当三维情形，最好并行来组装
        """


