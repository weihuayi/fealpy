import numpy as np

from .function import FiniteElementFunction
from ..common import ranges

class CPLFEMDof1d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p 
        self.multiIndex = self.multi_index_matrix() 
        self.cell2dof = self.cell_to_dof()

    def multi_index_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex 

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        if p == 1:
            return cell
        else:
            N = mesh.number_of_points()
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cell2dof = np.zeros((NC, ldof), dtype=np.int)
            cell2dof[:, [0, -1]] = cell
            cell2dof[:, 1:-1] = N + np.arange(NC*(p-1)).reshape(NC, p-1)
            return cell2dof

    def number_of_local_dofs(self):
        return self.p + 1

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        gdof = mesh.number_of_points()
        if p > 1:
            NC = mesh.number_of_cells()
            gdof += NC*(p-1)
        return gdof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point

        if p == 1:
            return point
        else:
            N = point.shape[0]
            gdof = self.number_of_global_dofs()
            ipoint = np.zeros(gdof, dtype=np.float)
            ipoint[:N] = point
            NC = mesh.number_of_cells()
            edge = mesh.ds.edge
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NC] = np.einsum('ij, kj->ki', w, point[cell]).reshape(-1)
            return ipoint

class CPLFEMDof2d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p 
        self.multiIndex = self.multi_index_matrix() 
        self.cell2dof = self.cell_to_dof()

    def multi_index_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs() 
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int)
        multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:,1] = idx0 - multiIndex[:,2]
        multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2] 
        return multiIndex

    def is_on_node_local_dof(self):
        isIdx = (self.multiIndex == self.p)
        isNodeDof = (isIdx[:, 0] | isIdx[:, 1] | isIdx[:, 2] | isIdx[:, 3]) 
        return isNodeDof

    def is_on_edge_local_dof(self):
        return self.multiIndex == 0 

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE= mesh.number_of_edges()
        N = mesh.number_of_points()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int) 
        edge2dof[:, [0, -1]] = edge 
        if p > 1:
            edge2dof[:, 1:-1] = N + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        if p == 1:
            cell2dof = cell

        if p > 1:
            cell2dof = np.zeros((NC, ldof), dtype=np.int)

            isEdgeDof = self.is_on_edge_local_dof()
            edge2dof = self.edge_to_dof()
            cell2edgeSign = mesh.ds.cell_to_edge_sign()
            cell2edge = mesh.ds.cell_to_edge()

            cell2dof[np.ix_(cell2edgeSign[:, 0], isEdgeDof[:, 0])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 0], [0]], :]
            cell2dof[np.ix_(~cell2edgeSign[:, 0], isEdgeDof[:,0])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 0], [0]], -1::-1]

            cell2dof[np.ix_(cell2edgeSign[:, 1], isEdgeDof[:, 1])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 1], [1]], -1::-1]
            cell2dof[np.ix_(~cell2edgeSign[:, 1], isEdgeDof[:,1])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 1], [1]], :]

            cell2dof[np.ix_(cell2edgeSign[:, 2], isEdgeDof[:, 2])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 2], [2]], :]
            cell2dof[np.ix_(~cell2edgeSign[:, 2], isEdgeDof[:,2])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 2], [2]], -1::-1]
        if p > 2:
            base = N + (p-1)*NE
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            idof = ldof - 3*p
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)
            
        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point

        if p == 1:
            return point
        if p > 1:
            N = point.shape[0]
            dim = point.shape[-1]
            gdof = self.number_of_global_dofs()
            ipoint = np.zeros((gdof, dim), dtype=np.float)
            ipoint[:N, :] = point
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1,2), dtype=np.float)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                    point[edge,:]).reshape(-1, dim)
        if p > 2:
            isEdgeDof = self.is_on_edge_local_dof()
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            w = self.multiIndex[isInCellDof, :]/p
            ipoint[N+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                    point[cell,:]).reshape(-1, dim)

        return ipoint  

    def number_of_global_dofs(self):
        p = self.p
        N = self.mesh.number_of_points()
        gdof = N
        if p > 1:
            NE = self.mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            ldof = self.number_of_local_dofs()
            NC = self.mesh.number_of_cells() 
            gdof += (ldof - 3*p)*NC 
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        return int((p+1)*(p+2)/2)

class CPLFEMDof3d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p 
        self.multiIndex = self.multi_index_matrix() 
        self.faceMultiIndex = self.face_multi_index_matrix()
        self.cell2dof = self.cell_to_dof()

    def multi_index_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        idx = np.arange(1, ldof)
        idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3) 
        idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
        idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
        idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
        multiIndex = np.zeros((ldof, 4), dtype=np.int)
        multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
        multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
        multiIndex[1:, 1] = idx0 - idx2
        multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
        return multiIndex

    def face_multi_index_matrix(self):
        p = self.p
        fdof = int((p+1)*(p+2)/2)
        idx = np.arange(0, fdof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        faceMultiIndex = np.zeros((fdof, 3), dtype=np.int)
        faceMultiIndex[:,2] = idx - idx0*(idx0 + 1)/2
        faceMultiIndex[:,1] = idx0 - faceMultiIndex[:,2]
        faceMultiIndex[:,0] = p - faceMultiIndex[:, 1] - faceMultiIndex[:, 2] 
        return faceMultiIndex

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = (self.multiIndex == p)
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        ldof = self.number_of_local_dofs()
        localEdge = self.mesh.ds.localEdge
        isEdgeDof = np.zeros((ldof, 6), dtype=np.bool)
        for i in range(6):
            isEdgeDof[i,:] = (self.multiIndex[localEdge[-(i+1), 0],:] == 0) & (self.multiIndex[localEdge[-(i+1), 1],:] == 0 )
        return isEdgeDof

    def is_on_face_local_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isFaceDof = (self.multiIndex == 0)
        return isFaceDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()

        base = N
        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int) 
        edge2dof[:, [0, -1]] = edge 
        if p > 1:
            edge2dof[:,1:-1] = base + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def face_to_dof(self):
        p = self.p
        fdof = int((p+1)*(p+2)/2)

        edgeIdx = np.zeros((2, p+1), dtype=np.int)
        edgeIdx[0, :] = range(p+1)
        edgeIdx[1, :] = edgeIdx[0, -1::-1]

        mesh = self.mesh

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()

        face = mesh.ds.face
        edge = mesh.ds.edge
        face2edge = mesh.ds.face_to_edge()

        edge2dof = self.edge_to_dof()

        face2dof = np.zeros((NF, fdof), dtype=np.int)
        faceIdx = self.faceMultiIndex
        isEdgeDof = (faceIdx == 0) 

        fe = np.array([1, 0, 0])
        for i in range(3):
            I = np.ones(NF, dtype=np.int)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2dof[:, isEdgeDof[:, i]] = edge2dof[face2edge[:, [i]], edgeIdx[I]]

        if p > 2:
            base = N + (p-1)*NE
            isInFaceDof = ~(isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2])
            fidof = fdof - 3*p
            face2dof[:, isInFaceDof] = base + np.arange(NF*fidof).reshape(NF, fidof)

        return face2dof

    def cell_to_dof(self):
        p = self.p
        fdof = int((p+1)*(p+2)/2)
        ldof = self.number_of_local_dofs()

        localFace = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

        mesh = self.mesh

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        face = mesh.ds.face 
        cell = mesh.ds.cell

        cell2face = mesh.ds.cell_to_face()

        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        face2dof = self.face_to_dof()
        isFaceDof = self.is_on_face_local_dof()
        faceIdx = self.faceMultiIndex.T

        for i in range(4):
            fi = face[cell2face[:, i]]
            fj = cell[:, localFace[i]]
            idxj = np.argsort(fj, axis=1)
            idxjr = np.argsort(idxj, axis=1)
            idxi = np.argsort(fi, axis=1)
            idx = idxi[np.arange(NC).reshape(-1, 1), idxjr]
            isCase0 = (np.sum(idx == np.array([1, 2, 0]), axis=1) == 3)
            isCase1 = (np.sum(idx == np.array([2, 0, 1]), axis=1) == 3)
            idx[isCase0, :] = [2, 0, 1]
            idx[isCase1, :] = [1, 2, 0]
            k = faceIdx[idx[:, 1], :] + faceIdx[idx[:, 2], :]
            a = (k*(k+1)/2 + faceIdx[idx[:, 2], :]).astype(np.int)
            cell2dof[:, isFaceDof[:, i]] = face2dof[cell2face[:, [i]], a]

        if p > 3:
            base = N + (p-1)*NE + (fdof - 3*p)*NF 
            idof = ldof - 4 - 6*(p - 1) - 4*(fdof - 3*p)
            isInCellDof = ~(isFaceDof[:, 0] | isFaceDof[:, 1] | isFaceDof[:, 2] | isFaceDof[:, 3])
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        N = mesh.number_of_points()
        gdof = N

        if p > 1:
            NE = mesh.number_of_edges()
            edof = p - 1
            gdof += edof*NE

        if p > 2:
            NF = mesh.number_of_faces()
            fdof = int((p+1)*(p+2)/2) - 3*p
            gdof += fdof*NF

        if p > 3:
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cdof = ldof - 6*edof - 4*fdof - 4
            gdof += cdof*NC

        return gdof


    def number_of_local_dofs(self):
        p = self.p
        ldof = (p+1)*(p+2)*(p+3)/6
        return int(ldof)

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point

        if p == 1:
            return point

        N = point.shape[0]
        dim = point.shape[1]
        NC = mesh.number_of_cells() 

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        ipoint = np.zeros((gdof, dim), dtype=np.float)
        ipoint[:N, :] = point
        if p > 1:
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1,2), dtype=np.float)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, kj...->ki...', w, point[edge,:]).reshape(-1, dim)
            
        if p > 2:
            NF = mesh.number_of_faces()
            fidof = int((p+1)*(p+2)/2) - 3*p
            face = mesh.ds.face
            isEdgeDof = (self.faceIdx == 0)
            isInFaceDof = ~(isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2])
            w = self.faceMultiIndex[isInFaceDof, :]/p
            ipoint[N+(p-1)*NE:N+(p-1)*NE+fidof*NF, :] = np.einsum('ij, kj...->ki...', w, point[face,:]).reshape(-1, dim)

        if p > 3:
            isFaceDof = self.is_on_face_local_dof()
            isInCellDof = ~(isFaceDof[:,0] | isFaceDof[:,1] | isFaceDof[:,2] | isFaceDof[:, 3])
            w = self.multiIndex[isInCellDof, :]/p
            ipoint[N+(p-1)*NE+fidof*NF:, :] = np.einsum('ij, kj...->ki...', w, point[cell,:]).reshape(-1, dim)
        return ipoint  


class DPLFEMDof1d():
    def __init__(self, mesh, p):
        pass

class DPLFEMDof2d():
    def __init__(self, mesh, p):
        pass

class DPLFEMDof3d():
    def __init__(self, mesh, p):
        pass


class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C'):
        self.mesh = mesh
        self.p = p 
        if spacetype is 'C':
            if mesh.meshType is 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.dim = 1
            if mesh.meshType is 'tri':
                self.dof = CPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif mesh.meshType is 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.dim = 3
        elif spacetype is 'D':
            if mesh.meshType is 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.dim = 1
            if mesh.meshType is 'tri':
                self.dof = DPLFEMDof2d(mesh, p) 
                self.dim = 2
            elif mesh.meshType is 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.dim = 3

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc 

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(dim+1,)` or `(nb, dim+1)`         

        Returns
        -------
        phi : numpy.array

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function
        dim = self.dim 
        multiIndex = self.dof.multiIndex 

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.linspace(0, 1, p, endpoint=False)
        shape = bc.shape[:-1]+(p+1, dim+1)
        A = np.ones(shape, dtype=np.float)
        A[..., 1:, :] = bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        phi = (p**p)*np.prod(A[..., multiIndex, [0, 1, 2]], axis=-1)
        return phi

    def grad_basis(self, bc):
        """
        compute the basis function values at barycentric point bc 

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(dim+1,)` or `(nb, dim+1)`         

        Returns
        -------
        phi : numpy.array

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function
        dim = self.dim 
        multiIndex = self.dof.multiIndex 

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)

        t = np.linspace(0, 1, p, endpoint=False)
        shape = bc.shape[:-1]+(p+1, dim+1)
        A = np.ones(shape, dtype=np.float)
        F = np.zeros(shape, dtype=np.float)
        A[..., 1:, :] = bc[..., np.newaxis, :] - t.reshape(-1, 1)
        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = 1
        np.cumprod(FF, axis=-2, out=FF)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)
        B = np.cumprod(A, axis=-2)
        B[..., 1:, :] *= P.reshape(-1, 1)

        Q = B[..., multiIndex, [0,1,2]]
        M = F[..., multiIndex, [0,1,2]]
        ldof = self.number_of_local_dofs()
        shape = bc.shape[:-1]+(ldof, dim+1)
        R = np.zeros(shape, dtype=np.float)

        for i in range(dim+1):
            idx = list(range(dim+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        pp = p**p
        Dlambda, _ = self.mesh.grad_lambda()
        gphi = np.einsum('...ij, kjm->...kim', pp*R, Dlambda)
        return gphi 

    def value(self, uh, bc):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        val = np.einsum('...j, ij->...i', phi, uh[cell2dof]) 
        return val 

    def grad_value(self, uh, bc):
        gphi = self.grad_basis(bc)
        cell2dof = self.dof.cell2dof
        val = np.einsum('...ijm, ij->...im', gphi, uh[cell2dof])
        return val

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass


    def interpolation(self, u):
        ipoint = self.dof.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=np.float)

class VectorLagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1):
        self.scalarspace = LagrangeFiniteElementSpace(mesh, p)

    def basis(self, bc):
        return self.scalarspace.basis(bc)

    def grad_basis(self, bc):
        return self.scalarspace.grad_basis(bc)

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        mesh = self.scalarspace.mesh
        NC = mesh.number_of_cells()
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        val = np.einsum('...j, ijm->...im', phi, uh[cell2dof])
        return val 

    def grad_value(self, uh, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gradphi = self.grad_basis(bc)
        cell2dof = self.scalarspace.dof.cell2dof()
        val = np.einsum('...ijm, ijk->...ikm', gradphi, uh[cell2dof])
        return val

    def div_value(self, uh, bc):
        val = self.grad_value(uh, bc)
        return np.diagonal(val, axis1=-2, axis2=-1) 

    def number_of_global_dofs(self):
        return self.scalarspace.number_of_global_dofs()
        
    def number_of_local_dofs(self):
        return self.scalarspace.number_of_local_dofs()

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof, self.scalarspace.dim), dtype=np.float)

