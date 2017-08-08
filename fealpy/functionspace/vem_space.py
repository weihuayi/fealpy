"""Virtual Element Space

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
    def __init__(self, mesh, p = 1, dtype=np.float):
        self.mesh = mesh
        self.barycenter = mesh.barycenter('cell')
        self.p = p

        area= mesh.area()
        self.area = area
        self.h = np.sqrt(area) 
        self.dtype = dtype

        self.cell_idx_matrix()

    def cell_idx_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs() 
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        self.cellIdx = np.zeros((ldof, 2), dtype=np.int)
        self.cellIdx[:,1] = idx - idx0*(idx0 + 1)/2
        self.cellIdx[:,0] = idx0 - cellIdx[:, 1]

    def degree_of_homogeneous(self, i):

        if i == 0:
            return self.cellIdx.sum(axis=1)
        elif i == 1:
            cellIdx = self.cellIdx
            ldof = self.number_of_local_dofs()
            r = cellIdx.sum(axis=1) 
            r[1:] -= 1
            r[np.min(cellIdx, axis=1) > 0] *= 2
            return r 
        elif i == 3:
            r = 0 #TODO:
            return r

    def basis(self, point):
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
        cellLocation = mesh.ds.cellLocation

        if p == 1:
            return cell, cellLocation
        else
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

