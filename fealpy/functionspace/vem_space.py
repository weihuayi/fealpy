"""
Virtual Element Space

"""
import numpy as np
from ..common import ranges
from .function import FiniteElementFunction

class MonomialSpace2d():
    def __init__(self, mesh, p = 0, dtype=np.float):
        self.mesh = mesh
        self.p = p

    def cell_idx_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs() 
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        self.cellIdx = np.zeros((ldof, 2), dtype=np.int)
        self.cellIdx[:,1] = idx - idx0*(idx0 + 1)/2
        self.cellIdx[:,0] = idx0 - cellIdx[:, 1]

    def number_of_local_dofs():
        p = self.p
        return int((p+1)*(p+2)/2)

    def basis(self, point):
        p = self.p
        cellIdx = self.cellIdx

        ldof = self.number_of_local_dofs() 
        NC = self.mesh.number_of_cells()
        phi = np.ones((NC, ldof), dtype=self.dtype)

        idx = np.sum(cellIdx, axis=1)
        idx0 = cellIdx[:, 0] 
        idx1 = cellIdx[:, 1]
        for i in range(1,ldof):
            phi[:, i] = (point[:, 0]**idx[i])*(point[:,1]**idx1[i])

        return phi

    def grad_basis(self, point):
        p = self.p
        cellIdx = self.cellIdx

        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs() 
        gradphi = np.zeros((NC, ldof, 2), dtype=self.dtype)

        idx = np.sum(cellIdx, axis=1)
        idx0 = cellIdx[:, 0] 
        idx1 = cellIdx[:, 1]

        for i in range(1, ldof):
            if cellIdx[i, 0] > 0:
                gradphi[:,i,0] = idx0[i]*point[:,0]**(idx0[i]-1)*point[:,1]**idx1[i]
            if cellIdx[i, 1] > 0:
                gradphi[:,i,1] = idx1[i]*(point[:,0]**idx0[i])*(point[:,1]**(idx1[i]-1))
        return gradphi

class ScaledMonomialSpace2d():
    def __init__(self, mesh, p=1, dtype=np.float):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.barycenter = mesh.barycenter('cell')
        self.p = p
        self.area= mesh.area()
        self.h = np.sqrt(self.area) 
        self.cellIdx = self.cell_idx_matrix()

        self.dtype = dtype

    def cell_idx_matrix(self):
        """
        Compute the natural correspondence from the one-dimensional index
        starting from 0. 

        Notes
        -----

        0<-->(0, 0), 1<-->(1, 0), 2<-->(0, 1), 3<-->(2, 0), 4<-->(1, 1),
        5<-->(0, 2), .....

        """
        p = self.p
        ldof = self.number_of_local_dofs() 
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        cellIdx = np.zeros((ldof, 2), dtype=np.int)
        cellIdx[:,1] = idx - idx0*(idx0 + 1)/2
        cellIdx[:,0] = idx0 - self.cellIdx[:, 1]
        return cellIdx

    def basis(self, point):
        """
        Compute the basis values at point

        Parameters
        ---------- 
        point : numpy array, NC x 2 
            The NC points on each cells
        """
        p = self.p
        h = self.h
        cellIdx = self.cellIdx

        ldof = self.number_of_local_dofs() 
        NC = self.mesh.number_of_cells()
        phi = np.ones((NC, ldof), dtype=self.dtype)

        v = point - self.barycenter
        idx = np.sum(cellIdx, axis=1)
        idx0 = cellIdx[:, 0] 
        idx1 = cellIdx[:, 1]
        for i in range(1,ldof):
            phi[:, i] = (v[:,0]**idx[i])*(v[:,1]**idx1[i])/(h**idx[i])
        return phi

    def value(self, uh, point):
        phi = self.basis(point)
        return np.einsum('ij, ij->i', uh, phi) 

    def grad_basis(self, point):
        p = self.p
        h = self.h
        cellIdx = self.cellIdx

        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs() 
        gradphi = np.zeros((NC, ldof, 2), dtype=self.dtype)

        v = point - self.barycenter
        idx = np.sum(cellIdx, axis=1)
        idx0 = cellIdx[:, 0] 
        idx1 = cellIdx[:, 1]
        for i in range(1, ldof):
            if cellIdx[i, 0] > 0:
                gradphi[:,i,0] = idx0[i]*v[:,0]**(idx0[i]-1)*v[:,1]**idx1[i]/(h**idx[i])
            if cellIdx[i, 1] > 0:
                gradphi[:,i,1] = idx1[i]*(v[:,0]**idx0[i])*(v[:,1]**(idx1[i]-1))/(h**idx[i])
        return gradphi

    def grad_value(self, uh, point):
        grad = self.grad_basis(point)
        return np.einsum('ij, ijm->im', uh, grad)

    def laplace_basis(self, point):
        p = self.p
        h = self.h
        cellIdx = self.cellIdx

        ldof = self.number_of_local_dofs() 
        NC = self.mesh.number_of_cells()
        lphi = np.ones((NC, ldof), dtype=self.dtype)

        v = point - self.barycenter
        idx = np.sum(cellIdx, axis=1)
        idx0 = cellIdx[:, 0] 
        idx1 = cellIdx[:, 1]
        
    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        return np.zeros((NC, ldof), dtype=self.dtype)

    def number_of_local_dofs(self):
        p = self.p
        return int((p+1)*(p+2)/2)

    def number_of_global_dofs(self):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        return NC*ldof

class VirtualElementSpace2d():
    def __init__(self, mesh, p = 1, dtype=np.float):
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, dtype)
        self.dtype = dtype
    
        self.B = self.V.get_matrix_B()
        self.D = self.V.get_matrix_D()

    def project_to_smspace(self, uh):
        S = self.smspace.function()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        idx = np.repeat(range(NC), NV)
        for i in range(3):
            S[:, i] = np.bincount(idx, weights=B[i, :]*uh[cell], minlength=NC)
        return S

    def get_matrix_B(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.h 
        cell2dof, cell2dofLocation = self.cell_to_dof()
        B = np.zeros((smldof, cell2dof.shape[0]), dtype=self.dtype) 
        if p==1:
            B[0, :] = 1/np.repeat(NV, NV)
            B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1,-1)
        else:
            raise ValueError("I have not code degree {} vem!".format(p))

        return B

    def get_matrix_G(self):
        p = self.p
        if p == 1:
            G = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        else:
            raise ValueError("I have not code degree {} vem!".format(p))
        return G

    def get_matrix_tilde_G(self):
        p = self.p 
        if p == 1:
            tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
        else:
            raise ValueError("I have not code degree {} vem!".format(p))
        return tG

    def get_matrix_D(self):
        p = self.p
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        bc = np.repeat(self.smspace.barycenter, NV, axis=0)
        D = np.ones((cell2dof.shape[0], smldof), dtype=self.dtype)
        D[:, 1:] = (point[cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
        return D

    def get_matrix_of_grad_projection(self):
        pass


    def basis(self, bc):
        pass

    def grad_basis(self, bc):
        pass

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass
    
    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        N = mesh.number_of_points()
        NE= mesh.number_of_edges()

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
        cellLocation = mesh.ds.cellLocation

        if p == 1:
            return cell, cellLocation
        else:
            ldof = self.number_of_local_dofs()
            NC = mesh.number_of_cells()
            NV = mesh.number_of_vertices_of_cells()

            cell2dofLocation = np.zeros(NC+1, dtype=np.int)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)
            cell2dof[ranges(NV) + np.repeat(cell2dofLocation[:-1], NV)] = cell
            
            edge2dof = self.edge_to_dof()
            cell2edgeSign = mesh.ds.cell_to_edge_sign()
            cell2edge = mesh.ds.cell_to_edge()

        return         

    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        N = mesh.number_of_points()
        gdof = N 
        p = self.p
        if p > 1:
            gdof += NE*(p-1) + NC*int((p-1)*p/2)
        return gdof

    def number_of_local_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        ldofs = NV
        p = self.p
        if p > 1:
            ldofs += NV*(p-1) + int((p-1)*p/2)
        return ldofs

    def interpolation_points(self):
        return self.mesh.point
    
    def function(self):
        return FiniteElementFunction(self)

    def interpolation(self, u):
        ipoint = self.interpolation_points()
        uI = self.function() 
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=self.dtype)

class VectorLagrangeFiniteElementSpace2d():
    def __init__(self, mesh, p=1, dtype=np.float):
        self.scalarspace = VirtualElementSpace2d(mesh, p, dtype=dtype)
        self.mesh = mesh
        self.p = p 
        self.dtype=dtype

    def basis(self, bc):
        pass

    def grad_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass

    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

    def number_of_global_dofs(self):
        return self.scalarspace.number_of_global_dofs()
        
    def number_of_local_dofs(self):
        return self.scalarspace.number_of_local_dofs()

    def cell_to_dof(self):
        return self.scalarspace.cell_to_dof()

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,2),dtype=self.dtype)
