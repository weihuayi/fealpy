import numpy as np
from .Mesh2d import Mesh2d, Mesh2dDataStructure


class QuadrangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(0, 1), (1, 2), (2, 3), (3, 0)])
    ccw = np.array([0, 1, 2, 3])
    V = 4
    E = 4
    F = 1

    def __init__(self, N, cell):
        super(QuadrangleMeshDataStructure, self).__init__(N, cell)


class QuadrangleMesh(Mesh2d):
    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = QuadrangleMeshDataStructure(NN, cell)

        self.meshtype = 'quad'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}

    def area(self, index=None):
        return self.cell_area(index=index)

    def cell_area(self, index=None):
        node = self.node
        cell = self.ds.cell
        if index is None:
            v1 = node[cell[:, 1], :] - node[cell[:, 0], :]
            v2 = node[cell[:, 2], :] - node[cell[:, 0], :]
        else:
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v1, v2)
        a0 = nv/2.0

        if index is None:
            v1 = node[cell[:, 3], :] - node[cell[:, 2], :]
            v2 = node[cell[:, 0], :] - node[cell[:, 2], :]
        else:
            v1 = node[cell[index, 3], :] - node[cell[index, 2], :]
            v2 = node[cell[index, 0], :] - node[cell[index, 2], :]
        nv = np.cross(v1, v2)
        a1 = nv/2.0
        return a0 + a1

    def uniform_refine(self, n=1):
        for i in range(n):
            N = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # Find the cutted edge  
            cell2edge = self.ds.cell_to_edge()
            edgeCenter = self.barycenter(entity='edge')
            cellCenter = self.barycenter(entity='cell')

            edge2center = np.arange(N, N+NE) 

            cell = self.ds.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(N + NE, N + NE + NC).reshape(-1, 1)
 
            cell = np.zeros((4*NC, 4), dtype=np.int)
            cell[0::4, :] = np.r_['1', cp[0], ep[0], cc, ep[3]] 
            cell[1::4, :] = np.r_['1', ep[0], cp[1], ep[1], cc]
            cell[2::4, :] = np.r_['1', cc, ep[1], cp[2], ep[2]]
            cell[3::4, :] = np.r_['1', ep[3], cc, ep[2], cp[3]]

            self.node = np.r_['0', self.node, edgeCenter, cellCenter]
            self.ds.reinit(N + NE + NC, cell)

        return 

    def angle(self):
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.ds.cell
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 4), dtype=np.float)
        iprev = [3, 0, 1, 2]
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            angle[:, i] = np.arccos(
                    np.sum(
                        v0*v1, axis=1
                    )/np.sqrt(
                        np.sum(v0**2, axis=1)*np.sum(v1**2, axis=1)))
        return angle

    def jacobi_at_corner(self):
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.entity('cell')
        localEdge = self.ds.local_edge()
        jacobi = np.zeros((NC, 4), dtype=np.float)
        iprev = [3, 0, 1, 2]
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            jacobi[:, i] = v0[:, 0]*v1[:, 1] - v0[:, 1]*v1[:, 0]
        return jacobi

    def cell_quality(self):
        jacobi = self.jacobi_at_corner()
        return jacobi.sum(axis=1)/4

    def bc_to_point(self, bc):
        node = self.node
        cell = self.ds.cell
        p = np.einsum('...j, ijk->...ik', bc, node[cell])
        return p 
