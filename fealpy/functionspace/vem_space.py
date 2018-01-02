"""
Virtual Element Space

"""
import numpy as np
from ..common import ranges
from .function import FiniteElementFunction
from ..quadrature import GaussLobattoQuadrature


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
        self.multiIndex = self.multi_index_matrix()

        self.dtype = dtype

    def get_matrix_H(self):
        p = self.p
        mesh = self.mesh
        point = mesh.point
        edge = mesh.ds.edge
        edge2cell = mesh.ds.edge2cell

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        ldof = self.number_of_local_dofs()

        H0 = np.zeros((NE, ldof, ldof), dtype=self.dtype)
        H1 = np.zeros((isInEdge.sum(), ldof, ldof), dtype=self.dtype)

        qf = GaussLobattoQuadrature(int(np.ceil((2*p+3)/2)))
        bcs, ws = qf.quadpts, qf.weights 
        ps = np.einsum('ij, kjm->ikm', bcs, point[edge])
        phi0 = self.basis(ps, edge2cell[:, 0])
        phi1 = self.basis(ps[:, isInEdge, :], edge2cell[isInEdge, 1])
        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1) 


        nm = mesh.edge_normal()
        b = point[edge[:, 0]] - self.barycenter[edge2cell[:, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = point[edge[isInEdge, 0]] - self.barycenter[edge2cell[isInEdge, 1]]
        H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        H = np.zeros((NC, ldof, ldof), dtype=self.dtype)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.multiIndex
        q = np.sum(multiIndex, axis=1)
        q = 1/(q + q.reshape(-1, 1) + 2)
        H *= q
        return H

    def multi_index_matrix(self):
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
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:,1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:,0] = idx0 - multiIndex[:, 1]
        return multiIndex 

    def basis(self, point, cellIdx=None):
        """
        Compute the basis values at point

        Parameters
        ---------- 
        point : numpy array
            The NC points on each cells
        """
        p = self.p
        h = self.h

        ldof = self.number_of_local_dofs() 
        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=self.dtype)
        if cellIdx is None:
            phi[..., 1:3] = (point - self.barycenter)/h.reshape(-1, 1)
        else:
            phi[..., 1:3] = (point - self.barycenter[cellIdx])/h[cellIdx].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    def value(self, uh, point):
        phi = self.basis(point)
        return np.einsum('ij, ij->i', uh, phi) 

    def grad_basis(self, point, cellIdx=None):
        p = self.p
        h = self.h
        ldof = self.number_of_local_dofs() 
        shape = point.shape[:-1]+(ldof, 2)
        gphi = np.zeros(shape, dtype=self.dtype)
        gphi[..., 1, 0] = 1 
        gphi[..., 2, 1] = 1
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            phi = self.basis(point, cellIdx)
            for i in range(2, p+1):
                gphi[..., start:start+i, 0] = np.einsum('i, ...i->...i', r[i-1::-1], phi[..., start-i:start])
                gphi[..., start+1:start+i+1, 1] = np.einsum('i, ...i->...i', r[0:i], phi[..., start-i:start])
                start += i+1
        if cellIdx is None:
            return gphi/h.reshape(-1, 1, 1)
        else:
            return gphi/h[cellIdx].reshape(-1, 1, 1)

    def grad_value(self, uh, point):
        grad = self.grad_basis(point)
        return np.einsum('ij, ijm->im', uh, grad)

    def laplace_basis(self, point, cellIdx=None):
        p = self.p
        h = self.h

        ldof = self.number_of_local_dofs() 

        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=self.dtype)
        if p > 1:
            lphi[:, 3:6:2] = 2
        if p > 2:
            start = 6
            r = np.r_[1, np.arange(1, p+1)]
            r = np.cumprod(r)
            r = r[2:]/r[0:-2]
            phi = self.basis(point, cellIdx)
            for i in range(3, p+1):
                lphi[..., start:start+i-1] += np.einsum('i, ...i->...i', r[i-2::-1], phi[..., start-2*i+1:start-i])
                lphi[..., start+i-1:start+i+1] += np.eisum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                start += i+1

        if cellIdx is None:
            return lphi/h.reshape(-1, 1)**2
        else:
            return lphi/h[cellIdx].reshape(-1, 1)**2
        
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
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()
    
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

    def get_matrix_D(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.h 
        cell2dof, cell2dofLocation = self.cell2dof, self.cell2dofLocation
        D = np.ones((cell2dof.shape[0], smldof), dtype=self.dtype)
        D[:, 1] = 

            

    def get_matrix_B(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.h 
        cell2dof, cell2dofLocation = self.cell2dof, self.cell2dofLocation
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
