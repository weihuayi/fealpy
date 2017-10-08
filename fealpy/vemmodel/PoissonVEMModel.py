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
        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell = mesh.ds.cell
        cellLocation = mesh.ds.cellLocation 
        barycenter = V.smspace.barycenter 

        B = self.B 
        h = V.smspace.h 
        area = V.smspace.area

        try:
            k = self.pde.diffusion_coefficient(barycenter)
        except  AttributeError:
            k = np.ones(NC) 

        S = np.zeros((NC, 3), dtype=self.dtype)
        idx = np.repeat(np.arange(NC), NV)
        for i in range(3):
            S[:, i] = np.bincount(idx, weights=B[i, :]*uh[cell], minlength=NC)
        S /=h.reshape(-1, 1)
        S *=k.reshape(-1, 1)
            

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

#        if 'subdomain' in dir(self.pde):
#            edge = mesh.ds.edge
#            edge2cell = mesh.ds.edge_to_cell()
#            isBdEdge = mesh.ds.boundary_edge_flag()
#            lp = barycenter[edge2cell[:, 0]]
#            rp = barycenter[edge2cell[:, 1]]
#            lidx = self.pde.subdomain(lp)
#            ridx = self.pde.subdomain(rp)
#            isSubDomainBdEdge = (lidx != ridx) & (~isBdEdge)
#            edge0 = edge[isSubDomainBdEdge]
#            edge2cell0 = edge2cell[isSubDomainBdEdge] 
#            n0 = mesh.edge_unit_normal(edgeflag=isSubDomainBdEdge)
#            ls = S[edge2cell0[:, 0], 1:3] - (S[edge2cell0[:, 0], 1:3]*n0).sum(axis=1, keepdims=True)*n0
#            rs = S[edge2cell0[:, 1], 1:3] - (S[edge2cell0[:, 1], 1:3]*n0).sum(axis=1, keepdims=True)*n0
#            if rtype is 'simple':
#                t = (ls + rs)/2
#            elif rtype is 'area':
#                larea = area[edge2cell0[:, 0]].reshape(-1, 1)
#                rarea = area[edge2cell0[:, 1]].reshape(-1, 1)
#                t = (larea*ls + rarea*rs)/(larea + rarea)
#            elif rtype is 'inv_area':
#                larea = 1/area[edge2cell0[:, 0]].reshape(-1, 1)
#                rarea = 1/area[edge2cell0[:, 1]].reshape(-1, 1)
#                t = (larea*ls + rarea*rs)/(larea + rarea)
#            else:
#                raise ValueError("I have note code method: {}!".format(rtype))
#
#            isSubDomainBdPoint = np.zeros(N, dtype=np.bool)
#            isSubDomainBdPoint[edge0] = True
#            idx0, = np.nonzero(isSubDomainBdPoint)
#            N0 = idx0.shape[0]
#            NE0 = edge0.shape[0]
#            idxMap = np.zeros(N, dtype=np.int)
#            idxMap[isSubDomainBdPoint] = np.arange(N0)
#
#            I = idxMap[edge0].flatten()
#            J = np.arange(NE0)
#            val = np.ones(2*NE0, dtype=np.bool)
#            p2e = csr_matrix((val, (I, np.repeat(J, 2))), shape=(N0, NE0), dtype=np.bool)
#            if rtype is 'simple':
#                d = p2e.sum(axis=1)
#                ruh[idx0, :] = p2e@t/d.reshape(-1, 1)
#            elif rtype is 'area':
#                length = mesh.edge_length()
#                d = p2e@length
#                ruh[idx0, :] = np.asarray((p2e@(t*length[isSubDomainBdEdge]))/d.reshape(-1, 1))
#            elif rtype is 'inv_area':
#                length = mesh.edge_length()
#                d = p2e@(1/length)
#                ruh[idx0, :] = np.asarray((p2e@(t/length[isSubDomainBdEdge].reshape(-1,1)))/d.reshape(-1, 1))
#            else:
#                raise ValueError("I have note code method: {}!".format(rtype))

        S1 = np.zeros((NC, 3), dtype=self.dtype)
        S2 = np.zeros((NC, 3), dtype=self.dtype)
        for i in range(3):
            S1[:, i] = np.bincount(idx, weights=B[i, :]*ruh[cell, 0], minlength=NC)
            S2[:, i] = np.bincount(idx, weights=B[i, :]*ruh[cell, 1], minlength=NC)

        point = mesh.point
        NV = mesh.number_of_vertices_of_cells()
        phi1 = (point[cell, 0] - np.repeat(barycenter[:, 0], NV))/np.repeat(h, NV)
        phi2 = (point[cell, 1] - np.repeat(barycenter[:, 1], NV))/np.repeat(h, NV)

        gx = np.repeat(S1[:, 0], NV)+ np.repeat(S1[:, 1], NV)*phi1 + \
            np.repeat(S1[:, 2], NV)*phi2 - np.repeat(S[:, 1], NV)
        gy = np.repeat(S2[:,  0], NV)+ np.repeat(S2[:, 1], NV)*phi1 + \
            np.repeat(S2[:, 2], NV)*phi2 - np.repeat(S[:, 2], NV)
        eta = np.sqrt(k*np.bincount(np.repeat(range(NC), NV), weights=gx**2+gy**2)/NV*area)
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

            BB = np.hsplit(B, cell2dofLocation[1:-1])
            DD = np.vsplit(D, cell2dofLocation[1:-1])
            cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

            
            f1 = lambda x: (x[1].T@G@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
            f2 = lambda x: np.repeat(x, x.shape[0]) 
            f3 = lambda x: np.tile(x, x.shape[0])
            f4 = lambda x: x.flatten()

            try:
                barycenter = V.smspace.barycenter 
                k = self.pde.diffusion_coefficient(barycenter)
            except  AttributeError:
                k = np.ones(NC) 

            K = list(map(f1, zip(DD, BB, k)))
            I = np.concatenate(list(map(f2, cd)))
            J = np.concatenate(list(map(f3, cd)))
            val = np.concatenate(list(map(f4, K)))
            A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=self.dtype)
        
        else:
            raise ValueError("I have not code the vem with degree > 1!")

        return A

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
