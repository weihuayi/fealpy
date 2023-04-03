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
        self.M = None # 需要组装的矩阵 
        self.atype = atype # 矩阵组装的方式，None、fast、ref
        self.dintegrators = [] # 区域积分子
        self.bintegrators = [] # 边界积分子

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
            return self.M@x
        else:
            out[:] = self.M@x

    def add_mult(self, x, y, a=1.0):
        y += a*(self.M@x)


    def assembly(self):
        """
        @brief 数值积分组装
        """
        space = self.space
        mesh = space[0].mesh

        if isinstance(space, tuple) and len(space) > 1:
            NC = mesh.number_of_cells() 
            GD = mesh.GD
            M = self.dintegrators[0].assembly_cell_matrix(space)
            c2f = space[0].dof.cell_to_dof()
            NN =mesh.number_of_nodes()
            #cell = mesh.entity('cell')
            #cell2dof = np.zeros((cell.shape[0], 2*GD), dtype=np.int_)
            
            if space0.doforder == 'vdims':
                for i in range(GD):
                    cell2dof[:, i::GD] = cell + NN*i
            
            elif space0.doforder == 'nodes':
                for i in range(GD):
                    cell2dof[:, i::GD] = cell*GD + i
            
            I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
            J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)
            self.M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN*GD, NN*GD))
        
        else:
            ldof = space.number_of_local_dofs()
            NC = mesh.number_of_cells() 
            M = np.zeros((NC, ldof, ldof), dtype=mesh.ftype)
            for inte in self.dintegrators:
                inte.assembly_cell_matrix(space, out=M)

            cell2dof = space.cell_to_dof()
            I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
            J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)

            gdof = space.number_of_global_dofs()
            self.M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))


    def fast_assembly(self):
        """
        @brief 免数值积分组装
        """

    def parallel_assembly(self):
        """
        @brief 多线程数值积分组装
        @note 特别当三维情形，最好并行来组装
        """


