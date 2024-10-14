import numpy as np
from numpy.linalg import inv, det
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vector_vem_space import VectorVirtualElementSpace2d 
from ..solver import solve
from ..boundarycondition import DirichletBC
from .integral_alg import PolygonMeshIntegralAlg
from ..quadrature import GaussLobattoQuadrature, GaussLegendreQuadrture 


class LinearElasticityVEMModel():
    def __init__(self, pde, mesh, p, q):
        self.p = p
        self.pde = pde
        self.mesh = mesh
        self.space = VectorVirtualElementSpace2d(mesh, p)
        self.integrator = mesh.integrator(q)
        self.area = self.space.vsmspace.area
        
        self.integralalg = PolygonMeshIntegralAlg(
           self.integrator, 
           mesh, 
           area=self.area, 
           barycenter=self.space.vsmspace.barycenter)

        self.uI = self.space.interpolation(pde.displacement, self.integralalg.integral)

    def matrix_G(self):
        p = self.p
        mesh = self.mesh
        mu = self.pde.mu
        lam = self.pde.lam

        def u0(x, cellidx): 
            sphi = self.space.vsmspace.strain_basis(x, cellidx=cellidx)
            dphi = self.space.vsmspace.div_basis(x, cellidx=cellidx)
            val = 2*mu*np.einsum('ijkmn, ijpmn->ijkp', sphi, sphi)
            val += lam*np.einsum('ijk, ijp->ijkp', dphi, dphi)
            return val 

        G = self.integralalg.integral(u0, celltype=True)

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()

        if p == 1:
            G[..., 0, 0] = 1
            G[..., 1, 1] = 1


            node = mesh.entity('node')
            bc = self.space.vsmspace.scalarspace.barycenter
            h = self.space.vsmspace.scalarspace.h
            cell = mesh.ds.cell

            idx = np.repeat(range(NC), NV)
            phi = (node[cell] - bc[idx])/np.repeat(h, NV).reshape(-1, 1)

            xx = phi[:, 0]**2
            yy = phi[:, 1]**2
            xy = phi[:, 0]*phi[:, 1]
            G[..., 3, :] = 0.0
            np.add.at(G[..., 3, 2], idx, -xy)  
            np.add.at(G[..., 3, 4], idx, -yy)
            np.add.at(G[..., 3, 3], idx, xx)
            G[..., 3, 2:5] /= NV.reshape(-1, 1)
            G[..., 3, 5] = -G[..., 3, 2] 
        elif p == 2:
            u = self.space.vsmspace.scalarspace.basis
            G0 = self.integralalg.integral(u, celltype=True)/self.area[:, np.newaxis]
            G[..., 0, 0::2] = G0
            G[..., 1, 1::2] = G0
            G[..., 3, :] = 0.0

            node = mesh.entity('node')
            edge = mesh.entity('edge')
            edge2cell = mesh.ds.edge_to_cell()
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

            qf = GaussLobattoQuadrature(p + 1)
            bcs = qf.quadpts
            ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
            phi = u(ps[0:-1], cellidx=edge2cell[:, 0])
            phi0 = np.sum(phi*phi[:, :, [2]], axis=0)
            np.add.at(G[..., 3, 0::2], edge2cell[:, 0], -phi0) 
            phi0 = np.sum(phi*phi[:, :, [1]], axis=0)
            np.add.at(G[..., 3, 1::2], edge2cell[:, 0], phi0)

            phi = u(ps[-1:0:-1, isInEdge], cellidx=edge2cell[isInEdge, 1])
            phi0 = np.sum(phi*phi[:, :, [2]], axis=0)
            np.add.at(G[..., 3, 0::2], edge2cell[isInEdge, 1], -phi0)
            phi0 = np.sum(phi*phi[:, :, [1]], axis=0)
            np.add.at(G[..., 3, 1::2], edge2cell[isInEdge, 1], phi0)
            G[..., 3, :] /= 2*NV.reshape(-1, 1)
            print(det(G[0]))
        else:
            def u1(x, cellidx):
                phip = self.space.vsmspace.scalarspace.basis(x, cellidx=cellidx)
                val = np.einsum('ijk, ijp->ijpk', phip, phip[..., 0:3])
                return val
            G0 = self.integralalg.integral(u1, celltype=True)/self.area[:, np.newaxis, np.newaxis]
            G[..., 0, 0::2] = G0[:, 0, :]
            G[..., 1, 1::2] = G0[:, 0, :]
            G[..., 3, 0::2] = -G0[:, 2, :]
            G[..., 3, 1::2] = G0[:, 1, :] 
        return G

    def matrix_H(self, p=None):
        if p is None:
            p = self.space.p
        mesh = self.mesh
        mu = self.pde.mu
        lam = self.pde.lam
        phi = self.space.vsmspace.basis

        def f(x, cellidx):
            phi = self.space.vsmspace.basis(x, cellidx=cellidx, p=p)
            val = np.einsum('ijkm, ijpm->ijkp', phi, phi)
            return val

        H = self.integralalg.integral(f, celltype=True)

        return H

    def matrix_B(self):
        p = self.space.p
        mesh = self.mesh
        mu = self.pde.mu
        lam = self.pde.lam

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        nm = mesh.edge_normal()

        qf = GaussLobattoQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights 
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])

        S0 = self.space.vsmspace.strain_basis(ps, cellidx=edge2cell[:, 0])
        val0 = np.einsum('i, jn, ijkmn->jkim', 2*mu*ws, nm, S0) 
        D0 = self.space.vsmspace.div_basis(ps, cellidx=edge2cell[:, 0])
        val0 += np.einsum('i, jn, ijk->jkin', lam*ws, nm, D0) 
        shape = val0.shape[:-2] + (-1, )
        val0 = val0.reshape(shape) 


        S1 = self.space.vsmspace.strain_basis(ps[-1::-1], cellidx=edge2cell[:, 1])
        val1 = np.einsum('i, jn, ijkmn->jkim', 2*mu*ws[-1::-1], -nm, S1)
        D1 = self.space.vsmspace.div_basis(ps[-1::-1], cellidx=edge2cell[:, 1])
        val1 += np.einsum('i, jn, ijk->jkin', lam*ws[-1::-1], -nm, D1)
        shape = val1.shape[:-2] + (-1, )
        val1 = val1.reshape(shape)

        cell2dof, cell2dofLocation = self.space.cell_to_dof() 
        smldof = self.space.vsmspace.number_of_local_dofs()
        B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float) 

        NV = mesh.number_of_vertices_of_cells()

        if p == 1:
            B[0, 0::2] = 1/np.repeat(NV, NV)
            B[1, 1::2] = B[0, 0::2] 

        elif p > 1:
            #TODO: correct for p >= 3?
            def u0(x, cellidx):
                val0 = self.space.vsmspace.div_strain_basis(x, cellidx=cellidx)
                val1 = self.space.vsmspace.grad_div_basis(x, cellidx=cellidx)
                val2 = self.space.vsmspace.basis(x, cellidx=cellidx, p=p-2)
                val = np.einsum('ijkm, ijpm->ijkp', 2*mu*val0 + lam*val1, val2)
                return val

            idx = (cell2dofLocation[0:-1] + 2*p*NV).reshape(-1, 1) + np.arange(p*(p-1))
            B0 = self.integralalg.integral(u0, celltype=True)
            H = inv(self.matrix_H(p=p-2))*self.area[..., np.newaxis, np.newaxis]
            B0 = B0@H
            B[:, idx] -= B0.swapaxes(0, 1) 

        # update the third line of B
        if p < 3:
            ipoints = self.space.interpolation_points()
            mask = np.zeros(cell2dof.shape[0], dtype=np.bool_)
            idx = cell2dofLocation[0:-1].reshape(-1, 1) + p*NV + np.arange(1:p
            mask[cell2dofLocation[

        else:


        NE = mesh.number_of_edges()
        for i in range(NE):
            idx0 = edge2cell[i, 0] 
            idx1 = cell2dofLocation[idx0] + (2*edge2cell[i, 2]*p + np.arange(2*(p+1)))%(2*NV[idx0]*p)
            B[:, idx1] += val0[i]
            if isInEdge[i]:
                idx0 = edge2cell[i, 1]
                idx1 = cell2dofLocation[idx0] + (2*edge2cell[i, 3]*p + np.arange(2*(p+1)))%(2*NV[idx0]*p)
                B[:, idx1] += val1[i]

        return B


    def matrix_D(self):
        p = self.space.p
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.space.cell_to_dof() 
        smldof = self.space.vsmspace.number_of_local_dofs()
        D = np.ones((len(cell2dof), smldof), dtype=np.float)

        if p == 1:
            bc = np.repeat(V.smspace.barycenter, NV, axis=0) 
            D[:, 1:] = (node[mesh.ds.cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            return D

        qf = GaussLobattoQuadrature(p+1)
        bcs, ws = qf.quadpts, qf.weights 
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = V.smspace.basis(ps[:-1], cellidx=edge2cell[:, 0])
        phi1 = V.smspace.basis(ps[p:0:-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)  
        D[idx, :] = phi0
        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            area = V.smspace.area
            idof = (p-1)*p//2 # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = H[:, :idof, :]/area.reshape(-1, 1, 1)
        return D
            

           






        
