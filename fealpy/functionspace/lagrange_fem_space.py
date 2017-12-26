import numpy as np

from .function import FiniteElementFunction
from ..common import ranges

class LagrangeFiniteElementSpace1d():
    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh = mesh
        self.p = p
        self.dtype= dtype

        self.cell_idx_matrix()

    def __str__(self):
        return "Lagrange finite element space on interval mesh!"

    def cell_idx_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        dim = self.mesh.geom_dimension()
        cellIdx = np.zeros((ldof, dim+1), dtype=np.int)
        cellIdx[:, 0] = np.arange(p, -1, -1)
        cellIdx[:, 1] = p - cellIdx[:, 0]
        self.cellIdx = cellIdx

    def basis(self, bc):

        mesh = self.mesh
        #dim = mesh.geom_dimension()
        dim = 1 

        p = self.p
        pp = p**p
        ldof = self.number_of_local_dofs() 
        phi = np.zeros((ldof, ), dtype=self.dtype)

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, dim+1), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t
        B = P.reshape(-1, 1)*np.multiply.accumulate(A, axis=0)
        phi = pp*np.prod(B[self.cellIdx, [0, 1]], axis=1)
        return phi


    def grad_basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()

        p = self.p
        pp = p**p
        ldof = self.number_of_local_dofs() 

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, dim+1), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t

        B = P.reshape(-1,1)*np.multiply.accumulate(A, axis=0)

        F = np.zeros((p+1, dim+1), dtype=self.dtype)
        for i in range(dim+1):
            Fi = A[1:, [i]] - np.diag(A[1:,i]) + np.eye(p)
            F[1:, i] = np.tril(np.multiply.accumulate(Fi, axis=0)).sum(axis=1)
        F *= P.reshape(-1,1)

        Dlambda, *_ = mesh.grad_lambda() 
        R = np.zeros((ldof, 2), dtype=np.float)

        Q = B[self.cellIdx,[0,1]]
        M = F[self.cellIdx,[0,1]]
        R[:,0] = M[:,0]*Q[:,1]
        R[:,1] = Q[:,0]*M[:,1]

        gradphi = np.einsum('ij,kj...->ki...', pp*R, Dlambda)
        return gradphi

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_points()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int)
        cell2dof[:, [0, -1]] = cell

        if p > 1:
            idof = p - 1
            cell2dof[:, 1:-1] = base + np.arange(NC*idof).reshape(NC, idof)
            
        return cell2dof

    def number_of_global_dofs(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        N = mesh.number_of_points()

        p = self.p

        gdof = N
        if p > 1:
            gdof += (p-1)*NC

        return gdof


    def number_of_local_dofs(self):
        p = self.p
        return p+1

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point

        NC = mesh.number_of_cells()
        N = mesh.number_of_points()

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()

        ipoint = np.zeros((gdof, ), dtype=np.float)
        ipoint[:N] = point
        if p > 1:
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1, 2), dtype=np.float)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NC] = np.einsum('ij, kj->ki', w, point[cell]).reshape(-1)
        return ipoint  


    def interpolation(self, u, uI):
        ipoint = self.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def array(self):
        pass

class LagrangeFiniteElementSpace2d():

    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh = mesh
        self.p = p 
        self.dtype=dtype
        self.cell_idx_matrix() 

    def __str__(self):
        return "Lagrange finite element space on triangle mesh!"

    def cell_idx_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs() 
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        cellIdx = np.zeros((ldof, 3), dtype=np.int)
        cellIdx[:,2] = idx - idx0*(idx0 + 1)/2
        cellIdx[:,1] = idx0 - cellIdx[:,2]
        cellIdx[:,0] = p - cellIdx[:, 1] - cellIdx[:, 2] 
        self.cellIdx = cellIdx

    def is_on_node_local_dof(self):
        p = self.p
        isIdx = (self.cellIdx == p)
        isNodeDof = (isIdx[:, 0] | isIdx[:, 1] | isIdx[:, 2] | isIdx[:, 3]) 
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        cellIdx = self.cellIdx
        ldof = self.number_of_local_dofs()
        return cellIdx == 0 

    def basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        # dim = mesh.geom_dimension()
        dim = 2

        pp = p**p
        ldof = self.number_of_local_dofs() 
        phi = np.zeros((ldof, ), dtype=self.dtype)

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, dim+1), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape(-1, 1)

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t
        B = P.reshape(-1, 1)*np.multiply.accumulate(A, axis=0)
        phi = pp*np.prod(B[self.cellIdx, [0, 1, 2]], axis=1)
        return phi

    def basis_einsum(self, bc):

        nb = len(bc) # the number of barrycenter points
        p = self.p   # the degree of polynomial basis function
        dim = 2

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)

        t = np.linspace(0, 1, p, endpoint=False).reshape(-1, 1)
        A = np.ones((nb, p+1, dim+1), dtype=np.float)
        A[:, 1:, :] = bc.reshape(-1, 1, 3) - t
        A[:, 1:, :] = np.cumprod(A[:, 1:, :], axis=1)
        A[:, 1:, :] = np.einsum('j, ijk->ijk', P, A[:, 1:, :])
        phi = (p**p)*np.prod(A[:, self.cellIdx, [0, 1, 2]], axis=2)
        return phi


    def grad_basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        #dim = mesh.geom_dimension()
        dim = 2

        pp = p**p
        ldof = self.number_of_local_dofs() 

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, dim+1), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t

        B = P.reshape(-1,1)*np.multiply.accumulate(A, axis=0)

        F = np.zeros((p+1, dim+1), dtype=self.dtype)
        for i in range(dim+1):
            Fi = A[1:, [i]] - np.diag(A[1:,i]) + np.eye(p)
            F[1:, i] = np.tril(np.multiply.accumulate(Fi, axis=0)).sum(axis=1)
        F *= P.reshape(-1,1)

        Dlambda, *_ = mesh.grad_lambda() 

        Q = B[self.cellIdx,[0,1,2]]
        M = F[self.cellIdx,[0,1,2]]
        R = np.zeros((ldof,3), dtype=np.float)
        R[:,0] = M[:,0]*Q[:,1]*Q[:,2]
        R[:,1] = Q[:,0]*M[:,1]*Q[:,2]
        R[:,2] = Q[:,0]*Q[:,1]*M[:,2]
        gradphi = np.einsum('ij,kj...->ki...', pp*R, Dlambda)
        return gradphi

    def grad_basis_einsum(self, bc):

        nb = len(bc) # the number of barrycenter points
        p = self.p   # the degree of polynomial basis function
        dim = 2

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)

        t = np.linspace(0, 1, p, endpoint=False).reshape(-1, 1)
        A = np.ones((nb, p+1, dim+1), dtype=np.float)
        A[:, 1:, :] = bc.reshape(-1, 1, 3) - t
        A[:, 1:, :] = np.cumprod(A[:, 1:, :], axis=1)
        A[:, 1:, :] = np.einsum('j, ijk->ijk', P, A[:, 1:, :])
        F = np.zeros((nb, p+1, dim+1), dtype=self.dtype)
        pass

    def value(self, uh, bc):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        return uh[cell2dof]@phi

    def grad_value(self, uh, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        NC = self.mesh.number_of_cells()
        val = np.zeros((NC, dim), dtype=self.dtype)
        val = np.einsum('ij, ij...->i...', uh[cell2dof], gradphi)
        return val

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

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

    def number_of_global_dofs(self):
        p = self.p
        N = self.mesh.number_of_points()
        ldof = self.number_of_local_dofs()
        gdof = N

        if p > 1:
            NE = self.mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            NC = self.mesh.number_of_cells() 
            gdof += (ldof - 3*p)*NC 

        return gdof

    def number_of_local_dofs(self):
        p = self.p
        return int((p+1)*(p+2)/2)

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point
        dim = mesh.geom_dimension()

        N = point.shape[0]
        dim = point.shape[1]
        NC = cell.shape[0]
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
            isEdgeDof = self.is_on_edge_local_dof()
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            w = self.cellIdx[isInCellDof, :]/p
            ipoint[N+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w, point[cell,:]).reshape(-1, dim)

        return ipoint  

    def interpolation(self, u):
        ipoint = self.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=self.dtype)

class VectorLagrangeFiniteElementSpace2d():
    def __init__(self, mesh, p=1, dtype=np.float):
        self.scalarspace = LagrangeFiniteElementSpace2d(mesh, p, dtype=dtype)
        self.mesh = mesh
        self.p = p 
        self.dtype=dtype

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
        val = np.zeros((NC,2), dtype=self.dtype)
        val[:,0] = (uh[cell2dof,0]@phi.reshape(-1, 1)).reshape(-1)
        val[:,1] = (uh[cell2dof,1]@phi.reshape(-1, 1)).reshape(-1)
        return val 

    def grad_value(self, uh, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        val = np.zeros((NC, 2, 2), dtype=self.dtype)
        val[:, 0, 0] = (uh[cell2dof, 0]*gradphi[:, :, 0]).sum(axis=1)
        val[:, 0, 1] = (uh[cell2dof, 0]*gradphi[:, :, 1]).sum(axis=1)
        val[:, 1, 0] = (uh[cell2dof, 1]*gradphi[:, :, 0]).sum(axis=1)
        val[:, 1, 1] = (uh[cell2dof, 1]*gradphi[:, :, 1]).sum(axis=1)
        return val

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        val = np.zeros((NC,), dtype=self.dtype)
        val += (uh[cell2dof, 0]*gradphi[:, :, 0]).sum(axis=1)
        val += (uh[cell2dof, 1]*gradphi[:, :, 1]).sum(axis=1)
        return val

    def number_of_global_dofs(self):
        return self.scalarspace.number_of_global_dofs()
        
    def number_of_local_dofs(self):
        return self.scalarspace.number_of_local_dofs()

    def cell_to_dof(self):
        return self.scalarspace.cell_to_dof()

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,2),dtype=self.dtype)

class LagrangeFiniteElementSpace3d():

    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh = mesh
        self.p = p 
        self.dtype=dtype

        self.cell_idx_matrix()
        self.face_idx_matrix()

    def __str__(self):
        return "Lagrange finite element space on tet mesh!"

    def cell_idx_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        idx = np.arange(1, ldof)
        idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3) 
        idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
        idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
        idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
        cellIdx = np.zeros((ldof, 4), dtype=np.int)
        cellIdx[1:, 3] = idx1 - idx2*(idx2 + 1)/2
        cellIdx[1:, 2] = idx2 - cellIdx[1:, 3]
        cellIdx[1:, 1] = idx0 - idx2
        cellIdx[:, 0] = p - np.sum(cellIdx[:, 1:], axis=1)
        self.cellIdx = cellIdx
        print(cellIdx)

    def face_idx_matrix(self):
        p = self.p
        fdof = int((p+1)*(p+2)/2)
        idx = np.arange(0, fdof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        faceIdx = np.zeros((fdof, 3), dtype=np.int)
        faceIdx[:,2] = idx - idx0*(idx0 + 1)/2
        faceIdx[:,1] = idx0 - faceIdx[:,2]
        faceIdx[:,0] = p - faceIdx[:, 1] - faceIdx[:, 2] 
        self.faceIdx = faceIdx
        print(faceIdx)

    def is_on_node_local_dof(self):
        p = self.p
        cellIdx = self.cellIdx
        isNodeDof = (cellIdx == p)
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        cellIdx = self.cellIdx
        ldof = self.number_of_local_dofs()
        localEdge = self.mesh.ds.localEdge
        isEdgeDof = np.zeros((ldof, 6), dtype=np.bool)
        for i in range(6):
            isEdgeDof[i,:] = (cellIdx[localEdge[-(i+1), 0],:] == 0)  & (cellIdx[localEdge[-(i+1), 1],:] == 0 )

        print('isEdgeDof', isEdgeDof)
        return isEdgeDof

    def is_on_face_local_dof(self):
        p = self.p
        cellIdx = self.cellIdx
        ldof = self.number_of_local_dofs()
        isFaceDof = (cellIdx == 0)
        return isFaceDof

    def basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()

        pp = p**p
        ldof = self.number_of_local_dofs() 
        phi = np.zeros((ldof, ), dtype=self.dtype)

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, dim + 1), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape(-1, 1)

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t
        B = P.reshape(-1, 1)*np.multiply.accumulate(A, axis=0)
        phi = pp*np.prod(B[self.cellIdx, [0, 1, 2, 3]], axis=1)
        return phi

    def grad_basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()

        pp = p**p
        ldof = self.number_of_local_dofs() 

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, dim+1), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t

        B = P.reshape(-1,1)*np.multiply.accumulate(A, axis=0)

        F = np.zeros((p+1, dim+1), dtype=self.dtype)
        for i in range(dim+1):
            Fi = A[1:, [i]] - np.diag(A[1:,i]) + np.eye(p)
            F[1:, i] = np.tril(np.multiply.accumulate(Fi, axis=0)).sum(axis=1)
        F *= P.reshape(-1,1)

        Dlambda, *_ = mesh.grad_lambda() 

        Q = B[self.cellIdx,[0, 1, 2, 3]]
        M = F[self.cellIdx,[0, 1, 2, 3]]
        R = np.zeros((ldof, dim+1), dtype=np.float)
        R[:,0] = M[:,0]*Q[:,1]*Q[:,2]*Q[:,3]
        R[:,1] = Q[:,0]*M[:,1]*Q[:,2]*Q[:,3]
        R[:,2] = Q[:,0]*Q[:,1]*M[:,2]*Q[:,3]
        R[:,3] = Q[:,0]*Q[:,1]*Q[:,2]*M[:,3]

        gradphi = np.einsum('ij,kj...->ki...', pp*R, Dlambda)
        return gradphi

    def hessian_basis(self, bc):
        pass

    def value(self, uh, bc):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        return uh[cell2dof]@phi
    
    def grad_value(self, uh, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        NC = self.mesh.number_of_cells()
        val = np.zeros((NC, dim), dtype=self.dtype)
        val = np.einsum('ij, ij...->i...', uh[cell2dof], gradphi)
        return val

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass
    
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
        faceIdx = self.faceIdx

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
        faceIdx = self.faceIdx.T

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
            w = self.faceIdx[isInFaceDof, :]/p
            ipoint[N+(p-1)*NE:N+(p-1)*NE+fidof*NF, :] = np.einsum('ij, kj...->ki...', w, point[face,:]).reshape(-1, dim)

        if p > 3:
            isFaceDof = self.is_on_face_local_dof()
            isInCellDof = ~(isFaceDof[:,0] | isFaceDof[:,1] | isFaceDof[:,2] | isFaceDof[:, 3])
            w = self.cellIdx[isInCellDof, :]/p
            ipoint[N+(p-1)*NE+fidof*NF:, :] = np.einsum('ij, kj...->ki...', w, point[cell,:]).reshape(-1, dim)

        return ipoint  

    def interpolation(self, u, uI):
        ipoint = self.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def array(self):
        pass
