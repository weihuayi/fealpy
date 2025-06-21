import numpy as np
from .FunctionSpace import FiniteElementSpace
from .Function import FiniteElementFunction
from ..common import ranges

class LagrangeFiniteElementSpace2d(FiniteElementSpace):

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

        cellIdx = np.zeros((3, ldof), dtype=np.int)
        cellIdx[:,0] = np.repeat(range(p, -1, -1), range(1, p+2))
        cellIdx[:,2] = ranges(range(1,p+2)) 
        cellIdx[:,1] = p - cellIdx[:, 0] - cellIdx[:,2] 
        self.cellIdx = cellIdx

    def is_on_node_local_dof(self):
        p = self.p
        cellIdx = self.cellIdx
        isNodeDof = (cellIdx == p)
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        cellIdx = self.cellIdx
        ldof = self.number_of_local_dofs()
        return cellidx == 0 

    def basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()

        pp = p**p
        ldof = self.number_of_local_dofs() 
        phi = np.zeros((ldof, ), dtype=self.dtype)

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1,3), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t
        B = P.reshape((-1,1))*np.multiply.accumulate(A, axis=0)
        phi = pp*np.prod(B[self.cellIdx, [0, 1, 2]], axis=1)
        return phi


    def grad_basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimension()

        pp = p**p
        ldof = self.number_of_local_dofs() 

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1,3), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t

        B = P.reshape((-1,1))*np.multiply.accumulate(A, axis=0)

        F = np.zeros((p+1,3), dtype=self.dtype)
        F0 = A[1:, [0]] - np.diag(A[1:,0]) + np.eye(p)
        F1 = A[1:, [1]] - np.diag(A[1:,1]) + np.eye(p)
        F2 = A[1:, [2]] - np.diag(A[1:,2]) + np.eye(p)
        F[1:,0] = np.tril(np.multiply.accumulate(F0, axis=0)).sum(axis=1)
        F[1:,1] = np.tril(np.multiply.accumulate(F1, axis=0)).sum(axis=1)
        F[1:,2] = np.tril(np.multiply.accumulate(F2, axis=0)).sum(axis=1)
        F *= P.reshape((-1,1))
        Dlambda, *_ = mesh.grad_lambda() 

        Q = B[self.cellIdx,[0,1,2]]
        M = F[self.cellIdx,[0,1,2]]
        R = np.zeros((ldof,3), dtype=np.float)
        R[:,0] = M[:,0]*Q[:,1]*Q[:,2]
        R[:,1] = Q[:,0]*M[:,1]*Q[:,2]
        R[:,2] = Q[:,0]*Q[:,1]*M[:,2]
        gradphi = np.einsum('ij,kj...->ki...', pp*R, Dlambda)
        return gradphi

    def dual_basis(self, u):
        ipoint = self.interpolation_points()
        return u(ipoint)

    def value(self, uh, bc):
        phi = self.basis(bc)
        cell2dof = self.cell_to_dof()
        return (uh[cell2dof]@phi.reshape((-1,1))).reshape((-1,))

    def grad_value(self, uh, bc):
        gradphi = self.grad_basis(bc)
        cell2dof = self.cell_to_dof()
        NC = self.mesh.number_of_cells()
        val = np.zeros((NC,2), dtype=self.dtype)
        val[:,0]= (uh[cell2dof]*gradphi[:,:,0]).sum(axis=1)
        val[:,1]= (uh[cell2dof]*gradphi[:,:,1]).sum(axis=1)
        return val

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE= mesh.number_of_edges()
        N = mesh.number_of_nodes()

        base = N

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE,p+1), dtype=np.int) 
        edge2dof[:, [0, -1]] = edge 
        if p > 1:
            edge2dof[:,1:-1] = base + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC= mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        isEdgeDof = self.is_on_edge_local_dof()
        edge2dof = self.edge_to_dof()
        cell2edgeSign = mesh.ds.cell_to_edge_sign()
        for i in range(3):
            cell2dof[cell2edgeSign[:,[i]], isEdgeDof[:,i]] = edge2dof[cell2edge[:,[i]],:]
            cell2dof[~cell2edgeSign[:,[i]], isEdgeDof[:,i]] = edge2dof[~cell2edge[:,[i]],-1::-1]
        if p > 2:
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
            ipoint[N:N+(p-1)*NE, :] = np.tensordot(w, point[edge,:], axes=(1,1)).reshape((-1,dim), order='F')

        if p > 2:
            isEdgeDof = self.is_on_edge_local_dof()
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            w = idx[isInCellDof,:]/p
            ipoint[N+(p-1)*NE:, :] = np.tensordot(w, point[cell,:], axes=(1,1)).reshape((-1,dim), order='F')
        return ipoint  

    def interpolation(self, u):
        ipoint = self.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=self.dtype)
