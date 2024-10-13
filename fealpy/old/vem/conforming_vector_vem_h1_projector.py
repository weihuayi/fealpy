import numpy as np
from numpy.linalg import inv
from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.vem import ConformingScalarVEMH1Projector2d 
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d 
import ipdb
from fealpy.quadrature import GaussLobattoQuadrature, GaussLegendreQuadrature
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
        self.B = self.assembly_cell_right_hand_side(space) 
        self.G = self.assembly_cell_left_hand_side(space) 
        g = lambda x: inv(x[0])@x[1]
        return list(map(g, zip(self.G, self.B)))

    def assembly_cell_right_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装 H1 投影算子的右端矩阵

        @retrun B 列表 B[i] 代表第 i 个单元上 H1 投影右端矩阵
        """
        p = space.p
        mesh = space.mesh
        M = self.M
        M = M[:, :p*(p+1)//2, :p*(p+1)//2]
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
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            smphi = space.smspace.basis(ps, index=index, p=p-1)
            vmgphi = space.vmspace.grad_basis(ps,index=index,p= p)
            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            w = np.array([(0, -1), (1, 0)])
            nm = v@w 
            sl = np.sqrt(v[:, 0]**2+v[:, 1]**2) # 边长(NE,)

            B11 = np.einsum('ijkmn,jm,i->kjin',vmgphi, nm, ws, optimize=True)#(2*ldof,NV[I],NQ,2)
            idx = np.zeros((B11.shape[1:]), dtype=np.int_)
            idx1 = np.arange(0, NV[i]*2*p, 2*p).reshape(-1, 1) + np.arange(0, 2*p+1, 2)
            idx1[-1, -1] = 0
            idx[:, :, 0] = idx1
            idx[:, :, 1] = idx1+1

            
            BB11 = np.zeros((vmldof, ldof[i]), dtype=np.float64)
            np.add.at(BB11, (np.s_[:], idx), B11)
            if p==1:
                w = np.tile(ws, NV[i]) * np.repeat(sl, ws.shape[0])
                np.add.at(BB11[0], idx[:, :, 0].flatten(), w)
                np.add.at(BB11[vmldof//2], idx[:, :, 1].flatten(), w)

                B.append(BB11)
            else:
                B12 = np.einsum('ijk,jl,i->kjil', smphi, nm, ws, optimize=True)#(ldof,NV[I],NQ,2)
                BB12 = np.zeros((p*(p+1)//2, ldof[i]), dtype=np.float64)
                np.add.at(BB12, (np.s_[:], idx), B12)
                B12 = E[i, ...]@A[i, ...]@BB12
                B1 = BB11-B12

                B2 = E[i,...]@A[i]@M[i]@K[i]

                B3 = np.zeros(((p-1)*(p-2)//2, ldof[i]),dtype=np.float64)
                B3[np.arange((p-1)*(p-2)//2), 2*p*NV[i]+np.arange((p-1)*(p-2)//2)] = cellarea[i] * 1
                B3 = E[i,...]@-(C@B3)
                
                BB = B1 + B2 + B3
                w = np.tile(ws, NV[i]) * np.repeat(sl, ws.shape[0])
                np.add.at(BB[0], idx[:, :, 0].flatten(), w)
                np.add.at(BB[vmldof//2], idx[:, :, 1].flatten(), w)

                B.append(BB)
        return B
                

    def assembly_cell_left_hand_side(self, space: ConformingVectorVESpace2d):
        """
        @brief 组装 H1 投影算子的左端矩阵

        @retrun G  G[i] 代表第 i 个单元上 H1 投影左端矩阵
        """
        p = space.p
        mesh = space.mesh
        M = self.M
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = space.mesh.number_of_cells()
        NV = space.mesh.number_of_vertices_of_cells()
        vmldof = space.vmspace.number_of_local_dofs()
        G = np.zeros((NC, vmldof, vmldof),dtype=np.float64)

        sspace = ConformingScalarVESpace2d(mesh, p) 
        d = ConformingVEMDoFIntegrator2d()
        D = d.assembly_cell_matrix(sspace, M)
        SH1Projector = ConformingScalarVEMH1Projector2d(D)
        G1 = SH1Projector.assembly_cell_left_hand_side(sspace)

        G[:, :vmldof//2, :vmldof//2] = G1
        G[:, -vmldof//2:, -vmldof//2:] = G1

        for i in range(NC):
            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 


            qf = GaussLegendreQuadrature(p+1) # NQ
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            vmphi = space.vmspace.basis(ps, index=index, p=p)
            v = node[cedge[:, 1]] - node[cedge[:, 0]]
            sl = np.sqrt(v[:, 0]**2+v[:, 1]**2) # 边长(NE,)
            G0 = np.einsum('ijkl,i,j->kl', vmphi, ws, sl, optimize=True) 

            G[i, 0, :] = G0[:, 0]
            G[i, vmldof//2, :] = G0[:, 1]
        return G 

