import numpy as np
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
import ipdb
from fealpy.quadrature import GaussLobattoQuadrature
from fealpy.vem.temporary_prepare import coefficient_of_div_VESpace_represented_by_SMSpace ,vector_decomposition, laplace_coefficient 


class ConformingVectorVEMH1Projector2d():
    def __init__(self, M):
        """
        @param M 质量矩阵，(n_{k-1}, n_{k-1})
        """
        self.M = M

    def assembly_cell_matrix(self, space: ConformingVectorVESpace2d):
        """
        @bfief 组装 H1 投影矩阵

        @return 返回为列表，列表中数组大小为
        """
        pass 

    def assembly_cell_right_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装 H1 投影算子的右端矩阵

        @retrun B 列表 B[i] 代表第 i 个单元上 H1 投影右端矩阵
        """
        p = space.p
        mesh = space.mesh
        M = self.M
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cellarea = mesh.cell_area()
        ldof = space.number_of_local_dofs()
        vmldof = space.vmspace.number_of_local_dofs()
        if p>1:
            A, C = vector_decomposition(space, p-2)
            K = coefficient_of_div_VESpace_represented_by_SMSpace(space, M)
        E = laplace_coefficient(space, p)
        B = []
        for i in range(NC):
            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 

            qf = GaussLobattoQuadrature(p + 1) # NQ
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge]) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            smphi = space.smspace.basis(ps, index=index, p=p-1)
            vmgphi = space.vmspace.grad_basis(ps,index=index,p= p)
            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            w = np.array([(0, -1), (1, 0)])
            nm = v@w 
            B11 = np.einsum('ijkmn,jm,i->kjin',vmgphi, nm, ws)#(2*ldof,NV[I],NQ,2)
            idx = np.zeros((B11.shape[1:]), dtype=np.int_)
            idx1 = np.arange(0, NV[i]*2*p, 2*p).reshape(-1, 1) + np.arange(0, 2*p+1, 2)
            idx1[-1, -1] = 0
            idx[:, :, 0] = idx1
            idx[:, :, 1] = idx1+1

            BB11 = np.zeros((vmldof, ldof[i]), dtype=np.float64)
            np.add.at(BB11, (np.s_[:], idx), B11)
            if p==1:
                B.append(BB11)
            else:
                B12 = np.einsum('ijk,jl,i->kjil', smphi, nm, ws)#(ldof,NV[I],NQ,2)
                BB12 = np.zeros((p*(p+1)//2, ldof[i]), dtype=np.float64)
                np.add.at(BB12, (np.s_[:], idx), B12)
                B12 = E[i, ...]@A[i, ...]@BB12
                B1 = BB11-B12

                B2 = E[i,...]@A[i]@M[i]@K[i]

                B3 = np.zeros(((p-1)*(p-2)//2, ldof[i]))
                B3[np.arange((p-1)*(p-2)//2), 2*p*NV[i]+np.arange((p-1)*(p-2)//2)] = cellarea[i] * 1
                B3 = E[i,...]@-(C@B3)
                
                BB = B1 + B2 + B3
                B.append(BB)
        return B
                


