import numpy as np
from numpy.linalg import inv
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
import ipdb
from fealpy.quadrature import GaussLobattoQuadrature, GaussLegendreQuadrature
from fealpy.vem.temporary_prepare import coefficient_of_div_VESpace_represented_by_SMSpace ,vector_decomposition, laplace_coefficient 

class ConformingVectorVEML2Projector2d():
    def __init__(self, M, PI1):
        self.M = M #(n_{k+1}, n_{k+1})
        self.PI1 = PI1


    def assembly_cell_matrix(self, space: ConformingVectorVESpace2d):
        C = self.assembly_cell_right_hand_side(space)
        H = self.assembly_cell_left_hand_side(space)
        g = lambda x: inv(x[0])@x[1]
        return list(map(g, zip(H, C)))


    def assembly_cell_right_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装 L2 投影算子的右端矩阵

        @retrun C 列表 C[i] 代表第 i 个单元上 L2 投影右端矩阵
        """
        p = space.p
        PI1 = self.PI1
        mesh = space.mesh
        M = self.M
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        cellarea = mesh.cell_area()
        vmldof = space.vmspace.number_of_local_dofs()
        ldof = space.number_of_local_dofs()
        A, J = vector_decomposition(space, p)
        K = coefficient_of_div_VESpace_represented_by_SMSpace(space, M[:, :p*(p+1)//2, :p*(p+1)//2])
        E = laplace_coefficient(space, p)
        C31 = self.integrator(space) 

        C = []
        for i in range(NC):
            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 
            qf = GaussLobattoQuadrature(p + 2) # NQ
            bcs, ws = qf.quadpts, qf.weights

            qf0 = GaussLobattoQuadrature(p+1)
            phi = space.edge_basis(qf0.quadpts[:, 0], qf.quadpts[:, 0])
            phi = np.tile(phi, (NV[i], 1, 1))

            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            smphi = space.smspace.basis(ps, index=index, p=p+1)
            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            w = np.array([(0, -1), (1, 0)])
            nm = v@w
            CC1 = np.einsum('ijk, jim, jl, i->kjml', smphi, phi, nm, ws, optimize=True)
            
            idx = np.zeros((NV[i], p+1, 2), dtype=np.int_)
            idx1 = np.arange(0, NV[i]*2*p, 2*p).reshape(-1, 1) + np.arange(0, 2*p+1, 2)
            idx1[-1, -1] = 0
            idx[:, :, 0] = idx1
            idx[:, :, 1] = idx1+1
            CCC1 = np.zeros(((p+2)*(p+3)//2, ldof[i]), dtype=np.float64)
            np.add.at(CCC1, (np.s_[:], idx), CC1)
            C1 = A[i]@CCC1

            C2 = A[i]@M[i, :, :p*(p+1)//2]@K[i]

            C3 = np.zeros((p*(p+1)//2, ldof[i]), dtype=np.float64)
            C3[:(p-1)*(p-2)//2, 2*p*NV[i]:2*p*NV[i]+(p-1)*(p-2)//2] = cellarea[i] * np.eye(((p-1)*(p-2)//2))
            C3[(p-2)*(p-1)//2:, :] = C31[i]@PI1[i]
            C3 = J@C3
            CC = C1-C2+C3
            C.append(CC) 
        return C 

    def integrator(self, space: ConformingVectorVESpace2d):

        """
        @brief 计算上面矩阵的第三部分
        """
        p = space.p
        mesh = space.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        vmldof = space.vmspace.number_of_local_dofs()
        C31 = np.zeros((NC, 2*p-1,vmldof),dtype=np.float64)
        for i in range(NC):
            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 

            qf = GaussLegendreQuadrature(p+2) # NQ
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            smphi = space.smspace.basis(ps, index=index, p=p-1)[:, :, (p-2)*(p-1)//2:]
            t = np.zeros((ws.shape[0], NV[i], 2))

            smphi1 = space.smspace.basis(ps, index=index, p=1)
            t[:, :, 0] = smphi1[:, :, 2]
            t[:, :, 1] = -smphi1[:, :, 1]

            vmphi = space.vmspace.basis(ps, index=index, p=p)
            H = np.einsum('ijl,ijk,ijml,i->jkm', t, smphi, vmphi, ws, optimize=True) #(NV[i], , 2*ldof)

            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            w = np.array([(0, -1), (1, 0)])
            nm = v@w 
            b = node[cedge[:, 0]] - mesh.entity_barycenter()[i]
            
            C31[i] = np.einsum('ijk,il,il->jk', H, nm, b, optimize=True)
            multiIndex = space.smspace.dof.multi_index_matrix(p=p)
            q = np.sum(multiIndex, axis=1)
            q = (q + q.reshape(-1, 1))[(p-2)*(p-1)//2:p*(p+1)//2, :] + 2 + 1
            C31[i] /= np.hstack((q,q)) 
              
        return C31
    def assembly_cell_left_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装 L2 投影算子的左端矩阵

        @retrun C 列表 C[i] 代表第 i 个单元上 L2 投影右端矩阵
        """
        p = space.p
        M = self.M         
        M = M[:, :(p+2)*(p+1)//2, :(p+2)*(p+1)//2]
        vmldof = space.vmspace.number_of_local_dofs()
        NC = space.mesh.number_of_cells()

        H = np.zeros((NC, vmldof, vmldof)) 
        for i in range(NC):
            H[i, :vmldof//2, :vmldof//2] = M[i]
            H[i, -vmldof//2:, -vmldof//2:] = M[i]
        return H


