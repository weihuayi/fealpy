import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

class PoissonVEMModel:
    def __init__(self, pde, V, dtype=np.float):
        self.V = V
        self.pde = pde  
        self.dtype=dtype

    def recover_estimate(self, uh, rtype='simple'):
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        cellLocation = mesh.ds.cellLocation 

        B = self.B 
        h = V.smspace.h 
        area = V.smspace.area

        S = np.zeros((NC, 3), dtype=self.dtype)
        idx = np.repeat(np.arange(NC), NV)
        for i in range(3):
            S[:, i] = np.bincount(idx, weights=B[i, :]*uh[cell], minlength=NC)
        S /=h.reshape(-1, 1)
            
        #for i in range(NC):
            #LB = B[:, cellLocation[i]:cellLocation[i+1]]
            #idx = cell[cellLocation[i]:cellLocation[i+1]]
            #S[i,:] = LB@uh[idx]/h[i]

        p2c = mesh.ds.point_to_cell()

        if rtype is 'simple':
            d = p2c.sum(axis=1)
            ruh = np.asarray((p2c@S[:, 1:3])/d.reshape(-1, 1))
        elif rtype is 'area':
            d = p2c@area
            ruh = np.asarray((p2c@(S[:, 1:3]*area.reshape(-1, 1)))/d.reshape(-1, 1))
        elif rtype is 'inv_area':
            d = p2c@(1/area)
            ruh = np.asarray((p2c@(S[:, 1:3]/area.reshape(-1,1)))/d.reshape(-1, 1))
        else:
            raise ValueError("I have note code method: {}!".format(rtype))

        S1 = np.zeros((NC, 3), dtype=self.dtype)
        S2 = np.zeros((NC, 3), dtype=self.dtype)
        for i in range(3):
            S1[:, i] = np.bincount(idx, weights=B[i, :]*ruh[cell, 0], minlength=NC)
            S2[:, i] = np.bincount(idx, weights=B[i, :]*ruh[cell, 1], minlength=NC)

        #for i in range(NC):
            #LB = B[:, cellLocation[i]:cellLocation[i+1]]
            #idx = cell[cellLocation[i]:cellLocation[i+1]]
            #S1[i, :] = (LB@ruh[idx, 0]).reshape(-1)
            #S2[i, :] = (LB@ruh[idx, 1]).reshape(-1)

        point = mesh.point
        barycenter = V.smspace.barycenter 
        NV = mesh.number_of_vertices_of_cells()
        phi1 = (point[cell, 0] - np.repeat(barycenter[:, 0], NV))/np.repeat(h, NV)
        phi2 = (point[cell, 1] - np.repeat(barycenter[:, 1], NV))/np.repeat(h, NV)

        gx = np.repeat(S1[:, 0], NV)+ np.repeat(S1[:, 1], NV)*phi1 + \
            np.repeat(S1[:, 2], NV)*phi2 - np.repeat(S[:, 1], NV)
        gy = np.repeat(S2[:,  0], NV)+ np.repeat(S2[:, 1], NV)*phi1 + \
            np.repeat(S2[:, 2], NV)*phi2 - np.repeat(S[:, 2], NV)
        eta = np.sqrt(np.bincount(np.repeat(range(NC), NV), weights=gx**2+gy**2)/NV*area)
        return eta

    def get_left_matrix(self):
        V = self.V
        mesh = V.mesh
        NC = mesh.number_of_cells()

        p = V.p
        area = V.smspace.area
        h = V.smspace.h

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        smldof = V.smspace.number_of_local_dofs()

        A = coo_matrix((gdof, gdof), dtype=self.dtype)

        cell2dof, cell2dofLocation = V.cell_to_dof()
        cell = mesh.ds.cell
        point = mesh.point

        if p == 1:
            NV = mesh.number_of_vertices_of_cells()
            B = np.zeros((smldof, cell2dof.shape[0]), dtype=self.dtype) 
            B[0, :] = 1/np.repeat(NV, NV)
            B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1,-1)
            self.B = B
            bc = np.repeat(V.smspace.barycenter, NV, axis=0)
            D = np.ones((cell2dof.shape[0], smldof), dtype=self.dtype)
            D[:, 1:] = (point[cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            G = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            for i in range(NC):
                K = B[:, cell2dofLocation[i]:cell2dofLocation[i+1]].T@G@B[:,
                        cell2dofLocation[i]:cell2dofLocation[i+1]]
                M = np.eye(ldof[i]) - D[cell2dofLocation[i]:cell2dofLocation[i+1], :]@B[:, cell2dofLocation[i]:cell2dofLocation[i+1]]
                K += M.T@M
                dof = cell[cell2dofLocation[i]:cell2dofLocation[i+1]]
                I = np.repeat(dof, ldof[i])
                J = np.repeat(dof.reshape(1, -1), ldof[i], axis=0).flatten()
                A += coo_matrix((K.flatten(), (I, J)), shape=(gdof, gdof), dtype=self.dtype)
        else:
            raise ValueError("I have not code the vem with degree > 1!")

        return A.tocsr()

    def get_right_vector(self):

        V = self.V
        mesh = V.mesh
        pde = self.pde

        ldof = V.number_of_local_dofs()
        bb = np.zeros(ldof.sum(), dtype=self.dtype)
        point = mesh.point
        NV = mesh.number_of_vertices_of_cells()
        F = pde.source(point)
        area = V.smspace.area
        cell2dof, cell2dofLocation = V.cell_to_dof()
        bb = F[cell2dof]/np.repeat(NV, NV)*np.repeat(area, NV)
        gdof = V.number_of_global_dofs()
        b = np.bincount(cell2dof, weights=bb, minlength=gdof)
        return b

    def get_neuman_vector(self):
        pass
