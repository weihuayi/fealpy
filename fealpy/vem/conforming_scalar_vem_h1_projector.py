import numpy as np

from ..functionspace import ConformingScalarVESpace2d
from ..functionspace import ConformingVectorVESpace2d


class ConformingScalarVEMH1Projector2d:

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d, M, D):
        pass

    def assembly_cell_righthand_side(self, space: ConformingScalarVESpace2d):
        """
        @brief 组装 H1 投影算子的右端矩阵

        @retrun B 列表 B[i] 代表第 i 个单元上 H1 投影右端矩阵
        """
        p = space.p
        mesh = space.mesh

        smldof = space.smspace.number_of_local_dofs()

        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.cellsize
        cell2dof = space.cell_to_dof() # 是一个自由度管理的列表
        B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float)
        if p == 1:
            B[0, :] = 1/np.repeat(NV, NV)
            B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1, -1)
            return B
        else:
            idx = cell2dofLocation[0:-1] + NV*p
            B[0, idx] = 1
            idof = (p-1)*p//2
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:]
            for i in range(2, p+1):
                idx0 = np.arange(start, start+i-1)
                idx1 =  np.arange(start-2*i+1, start-i)
                idx1 = idx.reshape(-1, 1) + idx1
                B[idx0, idx1] -= r[i-2::-1]
                B[idx0+2, idx1] -= r[0:i-1]
                start += i+1

            node = mesh.entity('node')
            edge = mesh.entity('edge')
            edge2cell = mesh.ds.edge_to_cell()
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

            qf = GaussLobattoQuadrature(p + 1)
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
            gphi0 = self.smspace.grad_basis(ps, index=edge2cell[:, 0])
            gphi1 = self.smspace.grad_basis(ps[-1::-1, isInEdge, :], index=edge2cell[isInEdge, 1])
            nm = mesh.edge_normal()

            # m: the scaled basis number,
            # j: the edge number,
            # i: the virtual element basis number

            NV = mesh.number_of_vertices_of_cells()

            val = np.einsum('i, ijmk, jk->mji', ws, gphi0, nm, optimize=True)
            idx = cell2dofLocation[edge2cell[:, [0]]] + \
                    (edge2cell[:, [2]]*p + np.arange(p+1))%(NV[edge2cell[:, [0]]]*p)
            np.add.at(B, (np.s_[:], idx), val)


            if isInEdge.sum() > 0:
                val = np.einsum('i, ijmk, jk->mji', ws, gphi1, -nm[isInEdge], optimize=True)
                idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
                        (edge2cell[isInEdge, 3].reshape(-1, 1)*p + np.arange(p+1)) \
                        %(NV[edge2cell[isInEdge, 1]].reshape(-1, 1)*p)
                np.add.at(B, (np.s_[:], idx), val)
            return B
