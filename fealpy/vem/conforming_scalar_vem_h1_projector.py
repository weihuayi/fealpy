import numpy as np
from fealpy.mesh.polygon_mesh import PolygonMesh
from fealpy.quadrature import GaussLobattoQuadrature

from fealpy.functionspace import ConformingScalarVESpace2d
from fealpy.functionspace import ConformingVectorVESpace2d
from fealpy.mesh import MeshFactory as MF
class ConformingScalarVEMH1Projector2d():
    def __init__(self):
        pass

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d, M, D):
        pass

    def assembly_cell_righthand_side(self, space: ConformingScalarVESpace2d):
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
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')
        BB = []
        for i in range(NC):
            smldof = space.smspace.number_of_local_dofs()
            ldof = space.number_of_local_dofs()
            B = np.zeros((smldof, ldof[i]),dtype=np.float_)
            if p==1:
                B[0, :] = 1/NV[i]
                B[1:, :] = mesh.node_normal()[i:i+NV,:].T/h[i]
                return B
            else:
                B[0,NV[i]*p] = 1 
                data = space.smspace.diff_index_2()
                xx = data['xx']
                yy = data['yy']
                B[xx[0],NV[i]*p+np.arange(xx[0].shape[0])] -= xx[1]
                B[yy[0],NV[i]*p+np.arange(yy[0].shape[0])] -= yy[1]

            cell2dof = mesh.entity('cell')[i]
            cell2edge = np.zeros((cell2dof.shape[0], 2),dtype=np.int_)
            cell2edge[:, 0] = cell2dof
            cell2edge[:-1, 1] = cell2dof[1:]
            cell2edge[-1, -1] = cell2dof[0]

            qf = GaussLobattoQuadrature(p + 1)
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cell2edge])
            gphi = space.smspace.grad_basis(ps, index=np.array([i]*ps.shape[1]))
            cell = mesh.entity('cell')
            v = node[cell2edge[:, 1],:] - node[cell2edge[:, 0],:]
            w = np.array([(0,-1),(1,0)])
            nm = v@w 
            val = np.einsum('i, ijmk, jk->mji', ws, gphi, nm, optimize=True)
            idx = np.arange(0,NV[i]*p,p).reshape(-1, 1)+np.arange(p+1)
            idx[-1,-1] = 0
            np.add.at(B, (np.s_[:], idx), val)
            BB.append(B)



        return BB





