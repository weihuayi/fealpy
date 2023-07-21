import numpy as np
from numpy.linalg import inv

from ..quadrature import GaussLegendreQuadrature
from ..functionspace import NonConformingScalarVESpace2d

class NonConformingScalarVEMH1Projector2d():
    def __init__(self, D):
        self.D = D

    def assembly_cell_matrix(self, space: NonConformingScalarVESpace2d):
        """
        @bfief 组装 H1 投影矩阵

        @return 返回为列表，列表中数组大小为(smldof,ldof)
        """
        p = space.p
        self.B = self.assembly_cell_right_hand_side(space) 
        self.G = self.assembly_cell_left_hand_side(space) 
        g = lambda x: inv(x[0])@x[1]
        return list(map(g, zip(self.G, self.B)))


    def assembly_cell_right_hand_side(self, space: NonConformingScalarVESpace2d):
        """
        @brief 组装 H1 投影算子的右端矩阵

        @retrun B 列表 B[i] 代表第 i 个单元上 H1 投影右端矩阵
        """

        p = space.p
        smldof = space.smspace.number_of_local_dofs()

        mesh = space.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dofLocation = space.dof.cell2dofLocation

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()

        B = np.zeros((smldof, cell2dofLocation[-1]), dtype=space.ftype)

        # the internal part
        if p > 1:
            NCE = mesh.number_of_edges_of_cells()
            idx = cell2dofLocation[0:-1] + NCE*p
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:]
            for i in range(2, p+1):
                idx0 = np.arange(start, start+i-1)
                idx1 = np.arange(start-2*i+1, start-i)
                idx1 = idx.reshape(-1, 1) + idx1
                B[idx0, idx1] -= r[i-2::-1]
                B[idx0+2, idx1] -= r[0:i-1]
                start += i+1

        # the normal deriveration part
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        gphi0 = space.smspace.grad_basis(ps, index=edge2cell[:, 0])
        gphi1 = space.smspace.grad_basis(
                ps[-1::-1, isInEdge, :],
                index=edge2cell[isInEdge, 1])
        nm = mesh.edge_normal()
        h = np.sqrt(np.sum(nm**2, axis=-1))
        # m: the scaled basis number,
        # j: the edge number,
        # i: the virtual element basis number
        val = np.einsum('i, ijmk, jk->mji', ws, gphi0, nm, optimize=True)
        idx = (cell2dofLocation[edge2cell[:, 0]]
                + edge2cell[:, 2]*p).reshape(-1, 1) + np.arange(p)
        B[:, idx] += val
        B[0, idx] = h.reshape(-1, 1)*ws

        val = np.einsum('i, ijmk, jk->mji', ws, gphi1, -nm[isInEdge], optimize=True)
        idx = ( cell2dofLocation[edge2cell[isInEdge, 1]]
                + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
        B[:, idx] += val
        B[0, idx] = h[isInEdge].reshape(-1, 1)*ws
        return np.hsplit(B, cell2dofLocation[1:-1]) 
    
    def assembly_cell_left_hand_side(self, space: NonConformingScalarVESpace2d):
        """
        @brief 组装 H1 投影算子的左端矩阵

        @return 列表 G[i] 代表第 i 个单元上 H1
        投影左端矩阵,数组大小为(smldof,smldof)
        """
        g = lambda x: x[0]@x[1]
        G = list(map(g, zip(self.B, self.D)))
        return G
