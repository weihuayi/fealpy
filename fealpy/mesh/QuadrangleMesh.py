import numpy as np
from .Mesh2d import Mesh2d, Mesh2dDataStructure

class QuadrangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(0,1),(1,2),(2,3),(3,0)])
    V = 4
    E = 4
    F = 1
    def __init__(self, N, cell):
        super(QuadrangleMeshDataStructure, self).__init__(N, cell)

class QuadrangleMesh(Mesh2d):
    def __init__(self, point, cell, dtype=np.float):
        self.point = point
        N = point.shape[0]
        self.ds = QuadrangleMeshDataStructure(N, cell)

        self.meshType = 'quad'
        self.dtype = dtype

    def area(self):
        NC = self.number_of_cells()
        point = self.point
        edge = self.ds.edge
        edge2cell = self.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        w = np.array([[0,-1],[1, 0]], dtype=np.int)
        v= (point[edge[:,1], :] - point[edge[:,0], :])@w
        val = np.sum(v*point[edge[:,0], :], axis=1)
        a = np.bincount(edge2cell[:,0], weights=val, minlength=NC)
        a+= np.bincount(edge2cell[isInEdge,1], weights=-val[isInEdge], minlength=NC)
        a /=2
        return a

    def angle(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 4), dtype=np.float)
        iprev = [3, 0, 1, 2]
        for i,j in localEdge:
            k = iprev[i] 
            v0 = point[cell[:,j],:] - point[cell[:,i],:]
            v1 = point[cell[:,k],:] - point[cell[:,i],:]
            angle[:,i] = np.arccos(np.sum(v0*v1,axis=1)/np.sqrt(np.sum(v0**2,axis=1)*np.sum(v1**2,axis=1)))
        return angle

    def jacobi_at_corner(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        localEdge = self.ds.local_edge()
        jacobi = np.zeros((NC, 4), dtype=np.float)
        iprev = [3, 0, 1, 2]
        for i,j in localEdge:
            k = iprev[i]
            v0 = point[cell[:,j],:] - point[cell[:,i],:]
            v1 = point[cell[:,k],:] - point[cell[:,i],:]
            jacobi[:,i] = v0[:,0]*v1[:,1] - v0[:,1]*v1[:,0]
        return jacobi

    def cell_quality(self):
        jacobi = self.jacobi_at_corner()
        return jacobi.sum(axis=1)/4

    def bc_to_point(self, bc):
        point = self.point
        cell = self.ds.cell
        p = np.einsum('...j, ijk->...ik', bc, point[cell])
        return p 
