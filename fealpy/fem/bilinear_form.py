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
        return self.assembly()

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
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis()


    def assembly_for_sspace_and_vspace_with_vector_basis(self) -> None:
        """
        组装标量空间（其基函数为标量函数）和向量空间（其基函数为向量函数）的矩阵的方法

        方法:
        1. 定义空间和获取局部和全局自由度数
        2. 获取网格和单元数量
        3. 初始化单元格矩阵 CM
        4. 对于每个区域积分器，组装单元矩阵
        5. 获取单元到自由度的映射
        6. 生成 I 和 J 的矩阵来构建稀疏矩阵
        7. 使用 CM、I 和 J 生成稀疏矩阵并保存到 _M 属性中

        """
        space = self.space
        ldof = space.dof.number_of_local_dofs()
        gdof = space.dof.number_of_global_dofs()

        mesh = space.mesh
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix(space, out=CM)

        cell2dof = space.dof.cell_to_dof()
        I = np.broadcast_to(cell2dof[:, :, None], shape=CM.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=CM.shape)
        self._M = csr_matrix((CM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        for bi in self.bintegrators:
            self._M += bi.assembly_face_matrix(space)

        return self._M

    def assembly_for_vspace_with_scalar_basis(self) -> None:
        """
        组装基函数由标量函数组合而成的向量函数空间的矩阵

        方法：
        1. 获取空间，确保其为元组类型，且其元素不为元组
        2. 获取网格、空间维度、局部和全局自由度数
        3. 从空间中获取单元到自由度的映射
        4. 获取网格的度量和单元数量
        5. 初始化单元矩阵 CM
        6. 对于每个区域积分器，组装单元矩阵
        7. 初始化稀疏矩阵 _M
        8. 根据空间的自由度排序优先级，进行对应的操作来更新 _M

        注意：这个函数不返回任何值，结果保存在 _M 属性中
        """
        space = self.space
        assert isinstance(space, tuple) and not isinstance(space[0], tuple)

        mesh = space[0].mesh
        GD = len(space) # 几个分量，几维问题
        ldof = space[0].number_of_local_dofs()
        gdof = space[0].number_of_global_dofs()
        cell2dof = space[0].cell_to_dof() # 标量空间的自由度矩阵
        
        cellmeasure = mesh.entity_measure()
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix(space, cellmeasure=cellmeasure, out=CM)
        self._M = csr_matrix((GD*gdof, GD*gdof), dtype=space[0].ftype)
        if space[0].doforder == 'sdofs': # 标量自由度排序优先
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
                        I = np.broadcast_to(cell2dof[:, :, None]+j*gdof, shape=val.shape)
                        J = np.broadcast_to(cell2dof[:, None, :]+i*gdof, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
        elif space[0].doforder == 'vdims': # 向量分量自由度排序优先
            for i in range(GD):
                for j in range(i, GD):
                    if i==j:
                        val = CM[:, i::GD, i::GD]
                        I = np.broadcast_to(GD*cell2dof[:, :, None] + i, shape=val.shape)
                        J = np.broadcast_to(GD*cell2dof[:, None, :] + i, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
                    else:
                        val = CM[:, i::GD, j::GD] 
                        I = np.broadcast_to(GD*cell2dof[:, :, None] + i, shape=val.shape)
                        J = np.broadcast_to(GD*cell2dof[:, None, :] + j, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))

                        self._M += csr_matrix((val.flat, (J.flat, I.flat)), shape=(GD*gdof, GD*gdof))

        for bi in self.bintegrators:
            self._M += bi.assembly_face_matrix(space)
        return self._M

    def quick_assembly(self):
        """
        @brief 数值积分组装

        @note space 可能是以下的情形
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.quick_assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.quick_assembly_for_sspace_and_vspace_with_vector_basis()

    def quick_assembly_for_sspace_and_vspace_with_vector_basis(self) -> None:
        """
        组装标量空间（其基函数为标量函数）和向量空间（其基函数为向量函数）的矩阵的方法

        方法:
        1. 定义空间和获取局部和全局自由度数
        2. 获取网格和单元数量
        3. 初始化单元格矩阵 CM
        4. 对于每个区域积分器，组装单元矩阵
        5. 获取单元到自由度的映射
        6. 生成 I 和 J 的矩阵来构建稀疏矩阵
        7. 使用 CM、I 和 J 生成稀疏矩阵并保存到 _M 属性中

        """
        space = self.space
        ldof = space.dof.number_of_local_dofs()
        gdof = space.dof.number_of_global_dofs()

        mesh = space.mesh
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix_quickly(space, out=CM)

        cell2dof = space.dof.cell_to_dof()
        I = np.broadcast_to(cell2dof[:, :, None], shape=CM.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=CM.shape)
        self._M = csr_matrix((CM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        for bi in self.bintegrators:
            self._M += bi.assembly_face_matrix(space)

        return self._M

    def quick_assembly_for_vspace_with_scalar_basis(self) -> None:
        """
        组装基函数由标量函数组合而成的向量函数空间的矩阵

        方法：
        1. 获取空间，确保其为元组类型，且其元素不为元组
        2. 获取网格、空间维度、局部和全局自由度数
        3. 从空间中获取单元到自由度的映射
        4. 获取网格的度量和单元数量
        5. 初始化单元矩阵 CM
        6. 对于每个区域积分器，组装单元矩阵
        7. 初始化稀疏矩阵 _M
        8. 根据空间的自由度排序优先级，进行对应的操作来更新 _M

        注意：这个函数不返回任何值，结果保存在 _M 属性中
        """
        space = self.space
        assert isinstance(space, tuple) and not isinstance(space[0], tuple)

        mesh = space[0].mesh
        GD = len(space) # 几个分量，几维问题
        ldof = space[0].number_of_local_dofs()
        gdof = space[0].number_of_global_dofs()
        cell2dof = space[0].cell_to_dof() # 标量空间的自由度矩阵
        
        cellmeasure = mesh.entity_measure()
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix_quickly(space, cellmeasure=cellmeasure, out=CM)
        self._M = csr_matrix((GD*gdof, GD*gdof), dtype=space[0].ftype)
        if space[0].doforder == 'sdofs': # 标量自由度排序优先
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
                        I = np.broadcast_to(cell2dof[:, :, None]+j*gdof, shape=val.shape)
                        J = np.broadcast_to(cell2dof[:, None, :]+i*gdof, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
        elif space[0].doforder == 'vdims': # 向量分量自由度排序优先
            for i in range(GD):
                for j in range(i, GD):
                    if i==j:
                        val = CM[:, i::GD, i::GD]
                        I = np.broadcast_to(GD*cell2dof[:, :, None] + i, shape=val.shape)
                        J = np.broadcast_to(GD*cell2dof[:, None, :] + i, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
                    else:
                        val = CM[:, i::GD, j::GD] 
                        I = np.broadcast_to(GD*cell2dof[:, :, None] + i, shape=val.shape)
                        J = np.broadcast_to(GD*cell2dof[:, None, :] + j, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))

                        self._M += csr_matrix((val.flat, (J.flat, I.flat)), shape=(GD*gdof, GD*gdof))

        for bi in self.bintegrators:
            self._M += bi.assembly_face_matrix(space)
        return self._M

    def fast_assembly(self, trialspace=None, testspace=None, coefspace=None):
        """
        @brief 免数值积分组装

        @note space 可能是以下的情形
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间组成的向量函数空间
            return self.fast_assembly_for_vspace_with_scalar_basis(trialspace, testspace, coefspace)
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.fast_assembly_for_sspace_and_vspace_with_vector_basis(trialspace, testspace, coefspace)

    def fast_assembly_for_sspace_and_vspace_with_vector_basis(self, trialspace, testspace, coefspace) -> None:
        """
        免数值积分组装标量空间（其基函数为标量函数）和向量空间（其基函数为向量函数）的矩阵的方法

        方法:
        1. 定义空间和获取局部和全局自由度数
        2. 获取网格和单元数量
        3. 初始化单元格矩阵 CM
        4. 对于每个区域积分器，组装单元矩阵
        5. 获取单元到自由度的映射
        6. 生成 I 和 J 的矩阵来构建稀疏矩阵
        7. 使用 CM、I 和 J 生成稀疏矩阵并保存到 _M 属性中

        """
        space = self.space
        trialspace = trialspace
        testspace = testspace
        coefspace = coefspace
        ldof = space.number_of_local_dofs()
        gdof = space.number_of_global_dofs()

        mesh = space.mesh
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix_fast(space, trialspace, testspace, coefspace, out=CM)

        cell2dof = space.cell_to_dof()
        I = np.broadcast_to(cell2dof[:, :, None], shape=CM.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=CM.shape)
        self._M = csr_matrix((CM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        for bi in self.bintegrators:
            self._M += bi.assembly_face_matrix(space)

        return self._M

    def fast_assembly_for_vspace_with_scalar_basis(self, trialspace, testspace, coefspace) -> None:
        """
        免数值积分组装基函数由标量函数组合而成的向量函数空间的矩阵

        方法：
        1. 获取空间，确保其为元组类型，且其元素不为元组
        2. 获取网格、空间维度、局部和全局自由度数
        3. 从空间中获取单元到自由度的映射
        4. 获取网格的度量和单元数量
        5. 初始化单元矩阵 CM
        6. 对于每个区域积分器，组装单元矩阵
        7. 初始化稀疏矩阵 _M
        8. 根据空间的自由度排序优先级，进行对应的操作来更新 _M

        注意：这个函数不返回任何值，结果保存在 _M 属性中
        """
        space = self.space
        assert isinstance(space, tuple) and not isinstance(space[0], tuple)

        mesh = space[0].mesh
        GD = len(space) # 几个分量，几维问题
        ldof = space[0].number_of_local_dofs()
        gdof = space[0].number_of_global_dofs()
        cell2dof = space[0].cell_to_dof() # 标量空间的自由度矩阵
        
        cellmeasure = mesh.entity_measure()
        NC = mesh.number_of_cells()
        CM = np.zeros((NC, GD*ldof, GD*ldof), dtype=space[0].ftype)
        for di in self.dintegrators:
            di.assembly_cell_matrix_fast(space, trialspace, testspace, coefspace, cellmeasure=cellmeasure, out=CM)
        self._M = csr_matrix((GD*gdof, GD*gdof), dtype=space[0].ftype)
        if space[0].doforder == 'sdofs': # 标量自由度排序优先
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
                        I = np.broadcast_to(cell2dof[:, :, None]+j*gdof, shape=val.shape)
                        J = np.broadcast_to(cell2dof[:, None, :]+i*gdof, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
        elif space[0].doforder == 'vdims': # 向量分量自由度排序优先
            for i in range(GD):
                for j in range(i, GD):
                    if i==j:
                        val = CM[:, i::GD, i::GD]
                        I = np.broadcast_to(GD*cell2dof[:, :, None] + i, shape=val.shape)
                        J = np.broadcast_to(GD*cell2dof[:, None, :] + i, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))
                    else:
                        val = CM[:, i::GD, j::GD] 
                        I = np.broadcast_to(GD*cell2dof[:, :, None] + i, shape=val.shape)
                        J = np.broadcast_to(GD*cell2dof[:, None, :] + j, shape=val.shape)
                        self._M += csr_matrix((val.flat, (I.flat, J.flat)), shape=(GD*gdof, GD*gdof))

                        self._M += csr_matrix((val.flat, (J.flat, I.flat)), shape=(GD*gdof, GD*gdof))

        for bi in self.bintegrators:
            self._M += bi.assembly_face_matrix(space)
        return self._M


    def parallel_assembly(self):
        """
        @brief 多线程数值积分组装
        @note 特别当三维情形，最好并行来组装
        """


