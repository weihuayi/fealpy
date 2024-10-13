import numpy as np
from scipy.sparse import csr_matrix

class LinearForm:
    """

    """
    def __init__(self, space, atype=None):
        """
        @brief 
        """
        self.space = space
        self._V = None # 需要组装的矩阵 
        self.atype = atype # 矩阵组装的方式，None、fast、ref
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

    def add_domain_integrator(self, I):
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

    def get_vector(self, copy=False):
        """
        @brief 获取线性型组装的向量

        @note 在关于时间的问题中
        """
        if copy is False:
            return self._V 
        else:
            return self._V.copy()

    def assembly(self):
        """
        @brief 数值积分组装

        @note space 可能是以下的情形, 程序上需要更好的设计
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间张成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis()

    def assembly_for_sspace_and_vspace_with_vector_basis(self):
        """
        @brief 基函数为标量函数的标量空间, 以及基函数为向量函数的函数空间
        """
        space = self.space
        mesh = space.mesh
        if mesh.meshtype == 'UniformMesh2d':
            NC = mesh.number_of_cells()
            cellmeasure = np.broadcast_to(mesh.entity_measure('cell'), (NC,))
        else:
            cellmeasure = mesh.entity_measure()

        NC = mesh.number_of_cells()
        gdof = space.dof.number_of_global_dofs()
        ldof = space.dof.number_of_local_dofs()

        bb = np.zeros((NC, ldof), dtype=space.ftype)
        for di in self.dintegrators:
            di.assembly_cell_vector(space, cellmeasure=cellmeasure, out=bb)

        cell2dof = space.dof.cell_to_dof()
        self._V = np.zeros((gdof, ), dtype=space.ftype)
        np.add.at(self._V, cell2dof, bb)

        for bi in self.bintegrators:
            bi.assembly_face_vector(space, out=self._V)

        return self._V

    def assembly_for_vspace_with_scalar_basis(self):
        """
        @brief 由标量空间张成的向量函数空间
        """
        space = self.space
        assert isinstance(space, tuple) and not isinstance(space[0], tuple)

        GD = space[0].geo_dimension()
        assert len(space) == GD

        mesh = space[0].mesh
        cellmeasure = mesh.entity_measure()

        NC = mesh.number_of_cells()
        gdof = space[0].number_of_global_dofs()
        ldof = space[0].number_of_local_dofs()

        cell2dof = space[0].cell_to_dof()
        if space[0].doforder == 'sdofs': # 标量空间自由度优先排序
            bb = np.zeros((NC, GD, ldof), dtype=mesh.ftype)
        elif space[0].doforder == 'vdims': # 向量分量自由度优先排序
            bb = np.zeros((NC, ldof, GD), dtype=mesh.ftype)

        for di in self.dintegrators:
            di.assembly_cell_vector(space, cellmeasure=cellmeasure, out=bb)

        self._V = np.zeros((GD*gdof, ), dtype=mesh.ftype)
        if space[0].doforder == 'sdofs': # 标量空间自由度优先排序
            V = self._V.reshape(GD, gdof)
            for i in range(GD):
                np.add.at(V[i, :], cell2dof, bb[:, i, :])
        elif space[0].doforder == 'vdims': # 向量分量自由度优先排序
            V = self._V.reshape(gdof, GD) 
            for i in range(GD):
                np.add.at(V[:, i], cell2dof, bb[:, :, i])
        
        for bi in self.bintegrators:
            bi.assembly_face_vector(space, out=self._V)

        return self._V

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()

