import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import triu, tril
from .Mesh2d import Mesh2d

class StructureQuadMesh(Mesh2d):
    def __init__(self, box, nx, ny, itype=np.int32, ftype=np.float):
        self.box = box
        self.ds = StructureQuadMeshDataStructure(nx, ny, itype)
        self.meshtype="quad"
        self.hx = (box[1] - box[0])/nx
        self.hy = (box[3] - box[2])/ny
        self.data = {}

        self.itype = itype 
        self.ftype = ftype 

    @property 
    def node(self):
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        box = self.box

        X, Y = np.mgrid[
                box[0]:box[1]:complex(0, nx+1), 
                box[2]:box[3]:complex(0, ny+1)]
        node = np.zeros((NN, 2), dtype=self.ftype)
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        return node

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_edges(self):
        return self.ds.NE

    def number_of_cells(self):
        return self.ds.NC

    def geo_dimension(self):
        return self.node.shape[1]

class StructureQuadMeshDataStructure:
    localEdge = np.array([(0,1),(1,2),(2,3),(3,0)])
    V = 4
    E = 4
    F = 1
    def __init__(self, nx, ny, itype):
        self.nx = nx
        self.ny = ny
        self.NN = (nx+1)*(ny+1)
        self.NE = ny*(nx+1) + nx*(ny+1)
        self.NC = nx*ny
        self.itype = itype

    
    @property
    def cell(self):

        nx = self.nx
        ny = self.ny

        NN = self.NN
        NC = self.NC
        cell = np.zeros((NC, 4), dtype=self.itype)
        idx = np.arange(NN).reshape(nx+1, ny+1)
        c = idx[:-1, :-1]
        cell[:, 0] = c.flat
        cell[:, 1] = cell[:, 0] + ny + 1
        cell[:, 2] = cell[:, 1] + 1
        cell[:, 3] = cell[:, 0] + 1
        return cell

    @property
    def edge(self):
        nx = self.nx
        ny = self.ny

        NN = self.NN
        NE = self.NE

        idx = np.arange(NN, dtype=self.itype).reshape(nx+1, ny+1)
        edge = np.zeros((NE, 2), dtype=self.itype)

        NE0 = 0
        NE1 = ny*(nx+1)
        edge[NE0:NE1, 0] = idx[:, :-1].flat
        edge[NE0:NE1, 1] = idx[:, 1:].flat
        edge[NE0:NE0+ny, :] = edge[NE0:NE0+ny, -1::-1]

        NE0 = NE1
        NE1 += nx*(ny+1)
        edge[NE0:NE1, 0] = idx[:-1, :].flat
        edge[NE0:NE1, 1] = idx[1:, :].flat
        edge[NE1:NE0:-nx-1, :] = edge[NE1:NE0:-nx-1, -1::-1]
        return edge

    @property
    def edge2cell(self):

        nx = self.nx
        ny = self.ny

        NC = self.NC
        NE = self.NE

        edge2cell = np.zeros((NE, 4), dtype=self.itype)

        idx = np.arange(NC).reshape(nx, ny).T

        # y direction
        idx0 = np.arange((nx+1)*ny, dtype=self.itype).reshape(nx+1, ny).T
        #left element
        edge2cell[idx0[:,1:], 0] = idx
        edge2cell[idx0[:,1:], 2] = 1
        edge2cell[idx0[:,0], 0] = idx[:,0]
        edge2cell[idx0[:,0], 2] = 3
        

        #right element
        edge2cell[idx0[:,:-1], 1] = idx
        edge2cell[idx0[:,:-1], 3] = 3
        edge2cell[idx0[:,-1], 1] = idx[:,-1]
        edge2cell[idx0[:,-1], 3] = 1

        # x direction 
        idx1 = np.arange(nx*(ny+1),dtype=self.itype).reshape(nx, ny+1).T
        NE0 = ny*(nx+1)
        #left element
        edge2cell[NE0+idx1[:-1], 0] = idx
        edge2cell[NE0+idx1[:-1], 2] = 0
        edge2cell[NE0+idx1[-1], 0] = idx[-1]
        edge2cell[NE0+idx1[-1], 2] = 2

        #right element
        edge2cell[NE0+idx1[1:], 1] = idx
        edge2cell[NE0+idx1[1:], 3] = 2
        edge2cell[NE0+idx1[0],1] = idx[0]
        edge2cell[NE0+idx1[0], 3] = 0

        return edge2cell

    def cell_to_node(self):
        """ 
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V*NC, dtype=np.bool)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool)
        return cell2node

    def cell_to_edge(self, sparse=False):
        """ The neighbor information of cell to edge
        """
        NE = self.NE
        NC = self.NC
        E = self.E

        edge2cell = self.edge2cell

        if sparse == False:
            cell2edge = np.zeros((NC, E), dtype=self.itype)
            cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = np.arange(NE,
                    dtype=self.itype)
            cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = np.arange(NE,
                    dtype=self.itype)
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

    def edge_to_node(self, sparse=False):
        NN = self.NN
        NE = self.NE

        edge = self.edge
        if sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flat
            val = np.ones(2*NE, dtype=np.bool)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
            return edge2node

    def edge_to_edge(self, sparse=False):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.tranpose()

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

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flat
        J = edge[:,[1,0]].flat
        val = np.ones((2*NE,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN),dtype=np.bool)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE
        
        edge = self.edge
        I = edge.flat
        J = np.repeat(range(NE), 2)
        val = np.ones(2*NE, dtype=np.bool)
        node2edge = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
        return node2edge

    def node_to_cell(self, localidx=False):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flat 
        J = np.repeat(range(NC), V)

        if localidx == True:
            val = ranges(V*np.ones(NC, dtype=np.int), start=1) 
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int)
        else:
            val = np.ones(V*NC, dtype=np.bool)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        return node2cell


    def boundary_node_flag(self):
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = np.zeros((N,), dtype=np.bool)
        isBdPoint[edge[isBdEdge,:]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    def boundary_cell_flag(self, bctype=None):
        """
        Parameters
        ----------
        bctype : None or 0, 1, 2 ,3
        """
        NC = self.NC

        if bctype is None:
            edge2cell = self.edge2cell
            isBdCell = np.zeros((NC,), dtype=np.bool)
            isBdEdge = self.boundary_edge_flag()
            isBdCell[edge2cell[isBdEdge,0]] = True

        else:
            cell2cell = self.cell_to_cell()
            isBdCell = cell2cell[:, bctype] == np.arange(NC)
        return isBdCell 

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx 

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx 

    def boundary_cell_index(self, bctype=None):
        isBdCell = self.boundary_cell_flag(bctype)
        idx, = np.nonzero(isBdCell)
        return idx 

    def y_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        return np.arange(ny*(nx+1))

    def x_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        return np.arange(ny*(nx+1), NE)

    def y_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        isYDEdge = np.zeros(NE, dtype=np.bool)
        isYDEdge[:ny*(nx+1)] = True
        return isYDEdge 

    def x_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        isXDEdge = np.zeros(NE, dtype=np.bool)
        isXDEdge[ny*(nx+1):] = True
        return isXDEdge  

    def left_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        return np.arange(ny+1)

    def right_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(NN-ny-1, NN)

    def bottom_boundary_node__index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN 
        return np.arange(0, NN-ny, ny+1)

    def up_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(ny, NN, ny+1)

    def peoriod_matrix(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        isPNode = np.zeros(NN, dtype=np.bool)
        lidx = self.left_boundary_node_index()
        ridx = self.right_boundary_node_index()
        bidx = self.bottom_boundary_node__index()
        uidx = self.up_boundary_node_index()

        isPNode[ridx] = True
        isPNode[uidx] = True
        NC = nx*ny

        val = np.ones(NC, dtype = np.bool)
        I = np.arange(NN)[~isPNode]
        J = range(NC) 
        C = coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)

        val = np.ones(nx, dtype=np.bool) 
        I = np.arange(NN)[uidx[:-1]]
        J = np.arange(0, NC-ny+1, ny)
        C += coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)

        val = np.ones(ny+1, dtype=np.bool)
        I = np.arange(NN)[ridx]
        J = np.arange(ny+1)
        J[-1] = 0
        C += coo_matrix((val,(I, J)), shape=(NN, NC), dtype=np.bool)

        return C

