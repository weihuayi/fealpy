import numpy as np
from numpy.linalg import inv

from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature

from ..functionspace import ConformingScalarVESpace2d

class ConformingScalarVEMH1Projector2d():
    def __init__(self, D):
        self.D = D

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d):
        """
        @bfief 组装 H1 投影矩阵

        @return 返回为列表，列表中数组大小为(smldof,ldof)
        """
        p = space.p
        self.B = self.assembly_cell_right_hand_side(space) 
        if p == 1:
            self.G = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
            return self.B
        else:
            self.G = self.assembly_cell_left_hand_side(space) 
            g = lambda x: inv(x[0])@x[1]
            return list(map(g, zip(self.G, self.B)))


    def assembly_cell_right_hand_side(self, space: ConformingScalarVESpace2d):
        """
        @brief 组装 H1 投影算子的右端矩阵

        @retrun B 列表 B[i] 代表第 i 个单元上 H1 投影右端矩阵
        """
        p = space.p
        mesh = space.mesh
        NC = mesh.number_of_cells()
        NV = mesh.ds.number_of_vertices_of_cells()
        h = space.smspace.cellsize
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        smldof = space.smspace.number_of_local_dofs()
        ldof = space.number_of_local_dofs()

        cell2dofLocation = space.dof.cell2dofLocation
        BB = np.zeros((smldof, cell2dofLocation[-1]), dtype=np.float64)
        B = np.hsplit(BB, cell2dofLocation[1:-1]) 

        for i in range(NC): #TODO：并行加加速
            if p==1:
                B[i][0, :] = 1/NV[i]
            else:
                B[i][0, NV[i]*p] = 1 
                data = space.smspace.diff_index_2()
                xx = data['xx']
                yy = data['yy']
                B[i][xx[0], NV[i]*p+np.arange(xx[0].shape[0])] -= xx[1]
                B[i][yy[0], NV[i]*p+np.arange(yy[0].shape[0])] -= yy[1]

            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 

            qf = GaussLobattoQuadrature(p + 1) # NQ
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge]) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            gphi = space.smspace.grad_basis(ps, index=index) # 求缩放基函数在每条边上的导函数值
            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            w = np.array([(0, -1), (1, 0)])
            nm = v@w 
            val = np.einsum('i, ijmk, jk->mji', ws, gphi, nm, optimize=True)
            idx = np.arange(0, NV[i]*p, p).reshape(-1, 1) + np.arange(p+1)
            idx[-1, -1] = 0
            np.add.at(B[i], (np.s_[:], idx), val)

        return B 
    
    def assembly_cell_left_hand_side(self, space: ConformingScalarVESpace2d):
        """
        @brief 组装 H1 投影算子的左端矩阵

        @return 列表 G[i] 代表第 i 个单元上 H1
        投影左端矩阵,数组大小为(smldof,smldof)
        """
        p = space.p
        mesh = space.mesh
        NC = mesh.number_of_cells()
        self.B = self.assembly_cell_right_hand_side(space) 

        if p == 1:
            G = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        else:
            g = lambda x: x[0]@x[1]
            G = list(map(g, zip(self.B, self.D))) #TODO： 并行加速
        return G
