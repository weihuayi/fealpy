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
            else:
                B[0,NV[i]*p] = 1 
                data = space.smspace.diff_index_2()
                xx = data['xx']
                yy = data['yy']
                B[xx[0],NV[i]*p+np.arange(xx[0].shape[0])] -= xx[1]
                B[yy[0],NV[i]*p+np.arange(yy[0].shape[0])] -= yy[1]

            cell2edge = np.zeros((cell[i].shape[0], 2),dtype=np.int_)
            cell2edge[:, 0] = cell[i]
            cell2edge[:-1, 1] = cell[i][1:]
            cell2edge[-1, -1] = cell[i][0]

            qf = GaussLobattoQuadrature(p + 1)
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cell2edge])
            gphi = space.smspace.grad_basis(ps, index=np.array([i]*ps.shape[1]))
            v = node[cell2edge[:, 1],:] - node[cell2edge[:, 0],:]
            w = np.array([(0,-1),(1,0)])
            nm = v@w 
            val = np.einsum('i, ijmk, jk->mji', ws, gphi, nm, optimize=True)
            idx = np.arange(0,NV[i]*p,p).reshape(-1, 1)+np.arange(p+1)
            idx[-1,-1] = 0
            np.add.at(B, (np.s_[:], idx), val)
            BB.append(B)
        return BB
    
    def assembly_cell_dof_matrix(self, space: ConformingScalarVESpace2d, H):
        """
        @brief 组装自由度矩阵 D

        @retrun DD 列表 DD[i] 代表第 i 个单元上的自由度矩阵
        """
        p = space.p
        mesh = space.mesh
        NC = mesh.number_of_cells()
        NV = mesh.ds.number_of_vertices_of_cells()
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        h = space.smspace.cellsize
        DD = []
        for i in range(NC):
            ldof = space.number_of_local_dofs()
            smldof = space.smspace.number_of_local_dofs()
            D = np.ones((ldof[i], smldof), dtype=np.float_)
            if p==1:
                bc = space.smspace.cellbarycenter[i]
                D[:, 1:] = (node[cell[i]]-bc)/h[i]
                DD.append(D)
            else:
                cell2edge = np.zeros((cell[i].shape[0], 2),dtype=np.int_)
                cell2edge[:, 0] = cell[i]
                cell2edge[:-1, 1] = cell[i][1:]
                cell2edge[-1, -1] = cell[i][0]

                qf = GaussLobattoQuadrature(p+1)
                bcs, ws = qf.quadpts, qf.weights
                ps = np.einsum('ij, kjm->ikm', bcs, node[cell2edge])
                phi = space.smspace.basis(ps[:-1],index=np.array([i]*ps.shape[1]))
                idx = np.arange(0, NV[i]*p, p)+np.arange(p).reshape(-1,1)
                D[idx, :] = phi
                
                area = space.smspace.cellmeasure
                idof = p*(p-1)//2
                idx = p*NV[i] +np.arange(idof)
                D[idx, :] = H[i,:idof,:]/area[i]
                DD.append(D)
        return DD
 



