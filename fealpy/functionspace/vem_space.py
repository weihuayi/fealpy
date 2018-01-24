"""
Virtual Element Space

"""
import numpy as np
from ..common import ranges
from .function import FiniteElementFunction

class SMDof2d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = self.multi_index_matrix()
        self.cell2dof = self.cell_to_dof()

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

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh

        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self):
        p = self.p
        return int((p+1)*(p+2)/2)

    def number_of_global_dofs(self):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        return NC*ldof


class ScaledMonomialSpace2d():
    def __init__(self, mesh, p=1):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.barycenter = mesh.barycenter('cell')
        self.p = p
        self.area= mesh.area()
        self.h = np.sqrt(self.area) 
        self.dof = SMDof2d(mesh, p)

    def basis(self, point, cellidx=None):
        """
        Compute the basis values at point

        Parameters
        ---------- 
        point : numpy array
            The shape of point is (..., M, 2) 
        """
        p = self.p
        h = self.h

        ldof = self.number_of_local_dofs() 
        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float) # (..., M, ldof)
        if cellidx is None:
            phi[..., 1:3] = (point - self.barycenter)/h.reshape(-1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            phi[..., 1:3] = (point - self.barycenter[cellidx])/h[cellidx].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    def value(self, uh, point, cellidx=None):
        phi = self.basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            return np.einsum('ij, ...ij->...i', uh[cell2dof], phi) 
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ij->...i', uh[cell2dof[cellidx]], phi)

    def grad_basis(self, point, cellidx=None):
        p = self.p
        h = self.h
        ldof = self.number_of_local_dofs() 
        shape = point.shape[:-1]+(ldof, 2)
        gphi = np.zeros(shape, dtype=np.float)
        gphi[..., 1, 0] = 1 
        gphi[..., 2, 1] = 1
        if p > 1:
            start = 3
            r = np.arange(1, p+1)
            phi = self.basis(point, cellidx=cellidx)
            for i in range(2, p+1):
                gphi[..., start:start+i, 0] = np.einsum('i, ...i->...i', r[i-1::-1], phi[..., start-i:start])
                gphi[..., start+1:start+i+1, 1] = np.einsum('i, ...i->...i', r[0:i], phi[..., start-i:start])
                start += i+1
        if cellidx is None:
            return gphi/h.reshape(-1, 1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            return gphi/h[cellidx].reshape(-1, 1, 1)

    def grad_value(self, uh, point, cellidx=None):
        gphi = self.grad_basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellidx is None:
            return np.einsum('ij, ijm->im', uh[cell2dof], gphi)
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ijm->...im', uh[cell2dof[cellidx]], gphi)

    def laplace_basis(self, point, cellidx=None):
        p = self.p
        area = self.area

        ldof = self.number_of_local_dofs() 

        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            start = 3
            r = np.r_[1, np.arange(1, p+1)]
            r = np.cumprod(r)
            r = r[2:]/r[0:-2]
            phi = self.basis(point, cellidx=cellidx)
            for i in range(2, p+1):
                lphi[..., start:start+i-1] += np.einsum('i, ...i->...i', r[i-2::-1], phi[..., start-2*i+1:start-i])
                lphi[..., start+2:start+i+1] += np.eisum('i, ...i->...i', r[0:i-1], phi[..., start-2*i+1:start-i])
                start += i+1

        if cellidx is None:
            return lphi/area.reshape(-1, 1)
        else:
            assert(point.shape[-2] == len(cellidx))
            return lphi/area[cellidx].reshape(-1, 1)
        
    def laplace_value(self, uh, point, cellidx=None):
        lphi = self.laplace_basis(point, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        if cellIdx is None:
            return np.einsum('ij, ...ij->...i', uh[cell2dof], lphi)
        else:
            assert(point.shape[-2] == len(cellidx))
            return np.einsum('ij, ...ij->...i', uh[cell2dof[cellIdx]], lphi)

    def function(self):
        return FiniteElementFunction(self)

    def array(self):
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        return np.zeros(NC*ldof, dtype=np.float)

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

class VEMDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

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
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = np.zeros(NC+1, dtype=np.int)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)
            
            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge2cell
            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
            cell2dof[idx] = edge2dof[:, 0:-1]
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

            idx = cell2dofLocation[edge2cell[isInEdge, [1]]] + edge2cell[isInEdge, [3]]*p + np.arange(p)
            cell2dof[idx] = edge2dof[isInEdge, -1:0:-1]

            NV = mesh.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = int((p-1)*p/2)
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + np.arange(int((p-1)*p/2))
            cell2dof[idx] = N + NE*(p-1) + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof, cell2dofLocation


    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        N = mesh.number_of_points()
        gdof = N 
        p = self.p
        if p > 1:
            gdof += NE*(p-1) + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        ldofs = NV
        p = self.p
        if p > 1:
            ldofs += NV*(p-1) + (p-1)*p//2
        return ldofs

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
            NE = mesh.number_of_edges()
            edof = N + NE*(p-1)
            ipoint = np.zeros((edof, dim), dtype=np.float)
            ipoint[:N, :] = point
            edge = mesh.ds.edge
            qf = GaussLobattoQuadrature(p + 1)
            bcs = qf.quadpts[1:-1, :]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', bcs, point[edge,:]).reshape(-1, dim)
            return ipoint

class VirtualElementSpace2d():
    def __init__(self, mesh, p = 1):
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.dof = VEMDof2d(mesh, p)

    def project_to_smspace(self, uh, B):
        #TODO: for general p  G^{-1}B
        S = self.smspace.function()
        NC = self.mesh.number_of_cells()
        NV = self.mesh.number_of_vertices_of_cells()
        idx = np.repeat(range(NC), NV)
        cell = self.mesh.ds.cell
        ldof = self.smspace.number_of_local_dofs()
        for i in range(ldof):
            S[i::ldof] = np.bincount(idx, weights=B[i, :]*uh[cell], minlength=NC)
        return S


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

    
    def function(self):
        return FiniteElementFunction(self)

    def interpolation(self, u):
        ipoint = self.interpolation_points()
        uI = self.function() 
        uI[:] = u(ipoint)
        return uI

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.mesh.point

    def projection(self, u, up):
        pass

    def array(self):
        gdof = self.number_of_global_dofs()
        return np.zeros((gdof,), dtype=np.float)

class VectorLagrangeFiniteElementSpace2d():
    def __init__(self, mesh, p=1):
        self.scalarspace = VirtualElementSpace2d(mesh, p)
        self.mesh = mesh
        self.p = p 

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
        return np.zeros((gdof,2), dtype=np.float)
