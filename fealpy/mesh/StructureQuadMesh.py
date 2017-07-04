import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import triu, tril

from matplotlib.collections import LineCollection

from .Mesh2d import Mesh2d

class StructureQuadMesh(Mesh2d):
    def __init__(self, box, nx, ny):
        self.box = box
        self.ds = StructureQuadMeshDataStructure(nx, ny)
        self.meshtype="quad"
        self.dx = (box[1] - box[0])/nx
        self.dy = (box[3] - box[2])/ny

    @property 
    def point(self):
        N = self.ds.N
        nx = self.ds.nx
        ny = self.ds.ny
        box = self.box

        X, Y = np.mgrid[
                box[0]:box[1]:complex(0, nx+1), 
                box[2]:box[3]:complex(0, ny+1)]
        point = np.zeros((N, 2), dtype=np.float)
        point[:, 0] = X.flatten()
        point[:, 1] = Y.flatten()
        return point


    def number_of_points(self):
        return self.ds.N

    def number_of_edges(self):
        return self.ds.NE

    def number_of_cells(self):
        return self.ds.NC

    def geom_dimension(self):
        return self.point.shape[1]

class StructureQuadMeshDataStructure:
    localEdge = np.array([(0,1),(1,2),(2,3),(3,0)])
    V = 4
    E = 4
    F = 1
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.N = (nx+1)*(ny+1)
        self.NE = ny*(nx+1) + nx*(ny+1)
        self.NC = nx*ny

    
    @property
    def cell(self):

        nx = self.nx
        ny = self.ny

        N = self.N
        NC = self.NC
        cell = np.zeros((NC, 4), dtype=np.int)
        idx = np.arange(N).reshape(nx+1, ny+1)
        c = idx[:-1, :-1]
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + ny + 1
        cell[:, 2] = cell[:, 1] + 1
        cell[:, 3] = cell[:, 0] + 1
        return cell

    @property
    def edge(self):
        nx = self.nx
        ny = self.ny

        N = self.N
        NE = self.NE

        idx = np.arange(N).reshape(nx+1, ny+1)
        edge = np.zeros((NE, 2), dtype=np.int)

        NE0 = 0
        NE1 = ny*(nx+1)
        edge[NE0:NE1, 0] = idx[:, :-1].flatten()
        edge[NE0:NE1, 1] = idx[:, 1:].flatten()
        edge[NE0:NE0+ny, :] = edge[NE0:NE0+ny, -1::-1]

        NE0 = NE1
        NE1 += nx*(ny+1)
        edge[NE0:NE1, 0] = idx[:-1, :].flatten('F')
        edge[NE0:NE1, 1] = idx[1:, :].flatten('F')
        edge[NE1-nx:NE1, :] = edge[NE1-nx:NE1, -1::-1]
        return edge

    @property
    def edge2cell(self):

        nx = self.nx
        ny = self.ny

        NC = self.NC
        NE = self.NE

        edge2cell = np.zeros((NE, 4), dtype=np.int)

        idx = np.arange(NC).reshape(nx, ny)

        # y direcion
        NE0 = 0 
        NE1 = ny
        edge2cell[NE0:NE1, 0] = idx[0]
        edge2cell[NE0:NE1, 1] = idx[0]
        edge2cell[NE0:NE1, 2:4] = 3

        NE0 = NE1
        NE1 += nx*ny
        edge2cell[NE0:NE1, 0] = idx.flatten()
        edge2cell[NE0:NE1, 2] = 1
        edge2cell[NE0:NE1-ny, 1] = idx[1:].flatten()
        edge2cell[NE0:NE1-ny, 3] = 3 
        edge2cell[NE1-ny:NE1, 1] = idx[-1]
        edge2cell[NE1-ny:NE1, 3] = 1

        # x direction 
        NE0 = NE1
        NE1 += nx
        edge2cell[NE0:NE1, 0] = idx[:, 0]
        edge2cell[NE0:NE1, 1] = idx[:, 0]
        edge2cell[NE0:NE1, 2:4] = 0

        NE0 = NE1
        NE1 += nx*ny
        edge2cell[NE0:NE1, 0] = idx.flatten('F')
        edge2cell[NE0:NE1, 2] = 3
        edge2cell[NE0:NE1-nx, 1] = idx[:, 1:].flatten('F')
        edge2cell[NE0:NE1-nx, 3] = 0 
        edge2cell[NE1-nx:NE1, 1] = idx[:, -1]
        edge2cell[NE1-nx:NE1, 3] = 3

        return edge2cell

    def cell_to_point(self):
        """ 
        """
        N = self.N
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V*NC, dtype=np.bool)
        cell2point = csr_matrix((val, (I, cell.flatten())), shape=(NC, N), dtype=np.bool)
        return cell2point

    def cell_to_edge(self, sparse=False):
        """ The neighbor information of cell to edge
        """
        NE = self.NE
        NC = self.NC
        E = self.E

        edge2cell = self.edge2cell

        if sparse == False:
            cell2edge = np.zeros((NC, E), dtype=np.int)
            cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = np.arange(NE)
            cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = np.arange(NE)
            return cell2edge
        else:
            val = np.ones(2*NE, dtype=np.bool)
            I = edge2cell[:, [0, 1]].flatten()
            J = np.repeat(range(NE), 2)
            cell2edge = csr_matrix(
                    (val, (I, J)), 
                    shape=(NC, NE), dtype=np.bool)
            return cell2edge 

    def cell_to_edge_sign(self, sparse=False):
        NC = self.NC
        E = self.E

        edge2cell = self.edge2cell
        if sparse == False:
            cell2edgeSign = np.zeros((NC, E), dtype=np.bool)
            cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True
        else:
            val = np.ones(NE, dtype=np.bool)
            cell2edgeSign = csr_matrix(
                    (val, (edge2cell[:, 0], range(NE))),
                    shape=(NC, NE), dtype=np.bool)
        return cell2edgeSign

    def cell_to_cell(self, return_sparse=False, return_boundary=True, return_array=False):
        """ Consctruct the neighbor information of cells
        """
        if return_array:                                                             
             return_sparse = False
             return_boundary = False
 
        NC = self.NC
        E = self.E
        edge2cell = self.edge2cell
        if (return_sparse == False) & (return_array == False):
            E = self.E
            cell2cell = np.zeros((NC, E), dtype=np.int)
            cell2cell[edge2cell[:, 0], edge2cell[:, 2]] = edge2cell[:, 1]
            cell2cell[edge2cell[:, 1], edge2cell[:, 3]] = edge2cell[:, 0]
            return cell2cell
        NE = self.NE
        val = np.ones((NE,), dtype=np.bool)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (edge2cell[:, 0], edge2cell[:, 1])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                    (val, (edge2cell[:, 1], edge2cell[:, 0])),
                    shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        else:
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            cell2cell = coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC+1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    def edge_to_point(self, sparse=False):
        N = self.N
        NE = self.NE

        edge = self.edge
        if sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flatten()
            val = np.ones(2*NE, dtype=np.bool)
            edge2point = csr_matrix((val, (I, J)), shape=(NE, N), dtype=np.bool)
            return edge2point

    def edge_to_edge(self, sparse=False):
        edge2point = self.edge_to_point()
        return edge2point*edge2point.tranpose()

    def edge_to_cell(self, sparse=False):
        if sparse==False:
            return self.edge2cell
        else:
            NC = self.NC
            NE = self.NE
            I = np.repeat(range(NF), 2)
            J = self.edge2cell[:, [0, 1]].flatten()
            val = np.ones(2*NE, dtype=np.bool)
            face2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool)
            return face2cell 

    def point_to_point(self):
        """ The neighbor information of points
        """
        N = self.N
        NE = self.NE
        edge = self.edge
        I = edge.flatten()
        J = edge[:,[1,0]].flatten()
        val = np.ones((2*NE,), dtype=np.bool)
        point2point = csr_matrix((val, (I, J)), shape=(N, N),dtype=np.bool)
        return point2point

    def point_to_edge(self):
        N = self.N
        NE = self.NE
        
        edge = self.edge
        I = edge.flatten()
        J = np.repeat(range(NE), 2)
        val = np.ones(2*NE, dtype=np.bool)
        point2edge = csr_matrix((val, (I, J)), shape=(NE, N), dtype=np.bool)
        return point2edge

    def point_to_cell(self, localidx=False):
        """
        """
        N = self.N
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flatten() 
        J = np.repeat(range(NC), V)

        if localidx == True:
            val = ranges(V*np.ones(NC, dtype=np.int), start=1) 
            point2cell = csr_matrix((val, (I, J)), shape=(N, NC), dtype=np.int)
        else:
            val = np.ones(V*NC, dtype=np.bool)
            point2cell = csr_matrix((val, (I, J)), shape=(N, NC), dtype=np.bool)
        return point2cell


    def boundary_point_flag(self):
        N = self.N
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = np.zeros((N,), dtype=np.bool)
        isBdPoint[edge[isBdEdge,:]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.NC
        edge2cell = self.edge2cell
        isBdCell = np.zeros((NC,), dtype=np.bool)
        isBdEdge = self.boundary_edge_flag()
        isBdCell[edge2cell[isBdEdge,0]] = True
        return isBdCell 

    def boundary_point_index(self):
        isBdPoint = self.boundary_point_flag()
        idx, = np.nonzero(isBdPoint)
        return idx 

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx 

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx 

