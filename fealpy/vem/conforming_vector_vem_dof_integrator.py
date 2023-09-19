import numpy as np
from fealpy.functionspace.conforming_vector_ve_space_2d import ConformingVectorVESpace2d
from fealpy.quadrature import GaussLobattoQuadrature,GaussLegendreQuadrature 
class ConformingVectorVEMDoFIntegrator2d:

    def assembly_cell_matrix(self, space: ConformingVectorVESpace2d):
        p = space.p
        mesh = space.mesh

        cell = mesh.entity('cell')
        node = mesh.entity('node')
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cellarea = mesh.cell_area()
        ldof = space.number_of_local_dofs()
        vmldof = space.vmspace.number_of_local_dofs()
        smldof = space.smspace.number_of_local_dofs()
        h = np.sqrt(mesh.cell_area())

        KK = []
        for i in range(NC):
            K = np.zeros((ldof[i], vmldof),dtype=np.float64)

            cedge = np.zeros((NV[i], 2), dtype=np.int_)
            cedge[:, 0] = cell[i]
            cedge[:-1, 1] = cell[i][1:]
            cedge[-1, -1] = cell[i][0] 

            qf = GaussLobattoQuadrature(p + 1) # NQ
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
            index = np.array([i]*NV[i]) 
            vmphi = space.vmspace.basis(ps, index=index, p=p)
            idx = np.arange(0, NV[i]*2*p, 2*p)+ np.arange(0, 2*p, 2).reshape(-1, 1) 
            K[idx] = vmphi[:-1, :, :, 0]
            K[idx+1] = vmphi[:-1, :, :, 1]
            #第二部分
            if p > 2:
                qf = GaussLegendreQuadrature(p+1) # NQ
                bcs, ws = qf.quadpts, qf.weights
                ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
                index = np.array([i]*NV[i]) 

                smphi = space.smspace.basis(ps, index=index, p=1)
                t = np.zeros((ws.shape[0], NV[i], 2),dtype=np.float64)
                t[..., 0] = smphi[..., 2]
                t[..., 1] = -smphi[..., 1]

                smphi = space.smspace.basis(ps, index=index, p=p-3)
                vmphi = space.vmspace.basis(ps, index=index, p=p)

                KK3 = np.einsum('ijkl,ijl,ijn,i->jnk',vmphi,t,smphi,ws, optimize=True)

                v = node[cedge[:, 1]] - node[cedge[:, 0]]
                w = np.array([(0, -1), (1, 0)])
                nm = v@w 
                b = node[cedge[:, 0]] - mesh.entity_barycenter()[i]
                
                K3 = np.einsum('ijk, il, il->jk', KK3, nm, b, optimize=True)

                multiIndex = space.smspace.dof.multi_index_matrix(p=p)
                q = np.sum(multiIndex, axis=1)
                q = (q + q.reshape(-1, 1) + 2 + 1)[:(p-1)*(p-2)//2,:]
                K3 /= np.hstack((q,q)) 
                K3 /= mesh.cell_area()[i]
                K[2*p*NV[i]:2*p*NV[i]+(p-1)*(p-2)//2, :] = K3
            #第四项
            if p > 1:
                qf = GaussLegendreQuadrature(p+1) # NQ
                bcs, ws = qf.quadpts, qf.weights
                ps = np.einsum('ij, kjm->ikm', bcs, node[cedge], optimize=True) # (NQ, NV[i], 2)
                index = np.array([i]*NV[i]) 
                smphi = space.smspace.basis(ps, index=index, p=p-1)[..., 1:]
                data = space.smspace.diff_index_1()
                x = data['x']
                y = data['y']
                smphi1 = space.smspace.basis(ps, index=index, p=p-1)
                dphi = np.concatenate((smphi1, smphi1), axis=2)
                c = np.hstack((x[1], y[1]))/h[i]

                KK4 = np.einsum('ijk,ijl,k,i->jlk',dphi, smphi, c, ws, optimize=True)

                v = node[cedge[:, 1]] - node[cedge[:, 0]]
                w = np.array([(0, -1), (1, 0)])
                nm = v@w 
                b = node[cedge[:, 0]] - mesh.entity_barycenter()[i]
                
                K4 = np.einsum('ijk, il, il->jk', KK4, nm, b, optimize=True)

                multiIndex = space.smspace.dof.multi_index_matrix(p=p-1)
                q = np.sum(multiIndex, axis=1)
                q = (q + q[1:].reshape(-1, 1) + 2)
                K4 /= np.hstack((q,q)) 
                K4 /= h[i]

                idx = np.hstack((x[0], y[0]+smldof))
                K[-((p)*(p+1)//2-1):, idx] = K4
            KK.append(K)
        return KK
     
