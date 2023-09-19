import numpy as np
from numpy.linalg import inv
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.quadrature import GaussLobattoQuadrature





class ConformingVectorVEMThirdProjector2d():
    def __init__(self, M, PI0):
        """
        @param M 质量矩阵，(n_k, n_k)
        @param PI0 L2投影矩阵，（2n_k, N_k）
        """
        self.M = M
        self.PI0 = PI0

    def assembly_cell_matrix(self, space: ConformingVectorVESpace2d):
        """
        @bfief 组装投影矩阵

        @return 返回为列表，列表中数组大小为

        """
        A = self.assembly_cell_right_hand_side(space)
        F = self.assembly_cell_left_hand_side(space)
        g = lambda x: inv(x[0])@x[1]
        return list(map(g,zip(F, A)))



    def assembly_cell_right_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装投影算子的右端矩阵

        @retrun  列表 A[i] 代表第 i 个单元上投影右端矩阵
        """
        p = space.p
        mesh = space.mesh
        M = self.M
        PI0 = self.PI0
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.entity('cell') 
        node = mesh.entity('node')
        h = np.sqrt(mesh.cell_area())
        smldof = space.smspace.number_of_local_dofs()
        ldof = space.number_of_local_dofs()
        M = M[:, :p*(p-1)//2, :smldof]
        data = space.smspace.diff_index_1(p=p-1)
        x = data['x']
        y = data['y']
        A = []
        for i in range(NC):
            A11 = np.zeros((p*(p+1)//2, smldof),dtype=np.float64)
            np.add.at(A11, (x[0],np.s_[:]), M[i]*(x[1]/h[i])[:, None])
            A12 = np.zeros((p*(p+1)//2, smldof),dtype=np.float64)
            np.add.at(A12, (y[0],np.s_[:]), M[i]*(y[1]/h[i])[:, None])
            A1 = np.zeros((2*p*(p+1), 2*smldof),dtype=np.float64)
            A1[:p*(p+1)//2, :smldof] = A11
            A1[p*(p+1)//2:p*(p+1), smldof:] = A11
            A1[p*(p+1):p*(p+1)*3//2, :smldof] = A12
            A1[p*(p+1)*3//2:p*(p+1)*2, smldof:] = A12

            A1 = -(A1@PI0[i])

            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 

            qf = GaussLobattoQuadrature(p + 1) # NQ
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            smphi = space.smspace.basis(ps, index=index, p=p-1)
            tmphi = np.zeros((ws.shape[0], NV[i], p*(p+1)*2, 2, 2),dtype=np.float64) 
            tmphi[:, :, :p*(p+1)//2, 0, 0] = smphi
            tmphi[:, :, p*(p+1)//2:p*(p+1), 0, 1] = smphi
            tmphi[:, :, p*(p+1):p*(p+1)*3//2, 1, 0] = smphi
            tmphi[:, :, p*(p+1)*3//2:p*(p+1)*2, 1, 1] = smphi

            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            w = np.array([(0, -1), (1, 0)])
            nm = v@w            

            AA2 = np.einsum('ijkmn,jm,i->kjin',tmphi, nm, ws, optimize=True)
            idx = np.zeros((NV[i], ws.shape[0], 2), dtype=np.int_)
            idx1 = np.arange(0, NV[i]*2*p, 2*p).reshape(-1, 1) + np.arange(0, 2*p+1, 2)
            idx1[-1, -1] = 0
            idx[:, :, 0] = idx1
            idx[:, :, 1] = idx1+1
            A2 = np.zeros((p*(p+1)*2, ldof[i]),dtype=np.float64)
            np.add.at(A2, (np.s_[:], idx), AA2)
            AA = A1 + A2
            A.append(AA)
        return A 
                

    def assembly_cell_left_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装 H1 投影算子的左端矩阵

        @retrun G 列表 G[i] 代表第 i 个单元上 H1 投影左端矩阵
        """                
        p = space.p
        mesh = space.mesh
        NC = mesh.number_of_cells()
        M = self.M
        smldof = space.smspace.number_of_local_dofs(p=p-1)
        M = M[:, :smldof, :smldof]
        F = np.zeros((NC, 4*smldof, 4*smldof),dtype=np.float64)
        F[:, :smldof, :smldof] = M
        F[:, smldof:2*smldof, smldof:2*smldof] = M
        F[:, 2*smldof:3*smldof, 2*smldof:3*smldof] = M
        F[:, 3*smldof:, 3*smldof:] = M
        return F




