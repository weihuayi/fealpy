import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from ..common import ranges

from ..quadrature import TriangleQuadrature
from ..quadrature import GaussLegendreQuadrature

from .Mesh2d import Mesh2d

class PolygonMesh(Mesh2d):

    """ 2d Polygon Mesh data structure from vtk data structure
    """
    def __init__(self, node, cell, cellLocation=None, topdata=None):
        self.node = node
        if cellLocation is None:
            if len(cell.shape)  == 2:
                NC = cell.shape[0]
                NV = cell.shape[1]
                cell = cell.reshape(-1)
                cellLocation = np.arange(0, (NC+1)*NV, NV)
            else:
                raise(ValueError("Miss `cellLocation` array!"))

        self.ds = PolygonMeshDataStructure(node.shape[0], cell, cellLocation,
                topdata=None)
        self.meshtype = 'polygon'
        self.itype = cell.dtype
        self.ftype = node.dtype

    def geo_dimension(self):
        return self.node.shape[-1]

    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(q)

    def number_of_vertices_of_cells(self):
        return self.ds.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.ds.number_of_edges_of_cells()

    def number_of_nodes_of_cells(self):
        return self.ds.number_of_vertices_of_cells()

    def to_vtk(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        cellLocation = self.ds.cellLocation
        NV = self.ds.number_of_vertices_of_cells()
        cells = np.zeros(len(cell) + NC, dtype=self.itype)
        isIdx = np.ones(len(cell) + NC, dtype=np.bool_)
        isIdx[0] = False
        isIdx[np.add.accumulate(NV+1)[:-1]] = False
        cells[~isIdx] = NV
        cells[isIdx] = cell
        return NC, cells

    @classmethod
    def from_mesh(cls, mesh):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        NV = cell.shape[1]
        cellLocation = np.arange(0, (NC+1)*NV, NV)
        return cls(node, cell.reshape(-1), cellLocation)

    @classmethod
    def from_quadtree(cls, quadtree):
        node, cell, cellLocation = quadtree.to_pmesh()
        return cls(node, cell, cellLocation)

    @classmethod
    def from_halfedgemesh(cls, mesh):
        """
        Notes
        -----
        从 HalfEdgeMesh2d 对象中拷贝出一个多边形网格出来。
        """
        node = mesh.entity('node')[:].copy()
        cell, cellLocation = mesh.entity('cell')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        return cls(node, cell, cellLocation, topdata=(edge, edge2cell))

    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.ds.cell, self.ds.cellLocation
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype='cell', index=None):
        node = self.node
        dim = self.geo_dimension()

        if etype in {'cell', 2}:
            cell2node = self.ds.cell_to_node()
            NV = self.number_of_vertices_of_cells().reshape(-1,1)
            bc = cell2node*node/NV
        elif etype in {'edge', 1}:
            edge = self.ds.edge
            bc = np.sum(node[edge, :], axis=1).reshape(-1, dim)/edge.shape[1]
        elif etype in {'node', 0}:
            bc = node
        return bc

    def angle(self):
        node = self.node
        cell = self.ds.cell
        cellLocation = self.ds.cellLocation

        idx1 = np.zeros(cell.shape[0], dtype=self.itype)
        idx2 = np.zeros(cell.shape[0], dtype=self.itype)

        idx1[0:-1] = cell[1:]
        idx1[cellLocation[1:]-1] = cell[cellLocation[:-1]]
        idx2[1:] = cell[0:-1]
        idx2[cellLocation[:-1]] = cell[cellLocation[1:]-1]
        a = node[idx1] - node[cell]
        b = node[idx2] - node[cell]
        la = np.sum(a**2, axis=1)
        lb = np.sum(b**2, axis=1)
        x = np.arccos(np.sum(a*b, axis=1)/np.sqrt(la*lb))
        return np.degrees(x)

    def node_normal(self):
        node = self.node
        cell, cellLocation = self.entity('cell')

        idx1 = np.zeros(cell.shape[0], dtype=self.itype)
        idx2 = np.zeros(cell.shape[0], dtype=self.itype)

        idx1[0:-1] = cell[1:]
        idx1[cellLocation[1:]-1] = cell[cellLocation[:-1]]
        idx2[1:] = cell[0:-1]
        idx2[cellLocation[:-1]] = cell[cellLocation[1:]-1]

        w = np.array([(0,-1),(1,0)])
        d = node[idx1] - node[idx2]
        return 0.5*d@w

    def area(self, index=None):
        #TODO: 3D Case
        NC = self.number_of_cells()
        node = self.node
        edge = self.ds.edge
        edge2cell = self.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        w = np.array([[0, -1], [1, 0]], dtype=self.itype)
        v= (node[edge[:, 1], :] - node[edge[:, 0], :])@w
        val = np.sum(v*node[edge[:, 0], :], axis=1)
        a = np.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a+= np.bincount(edge2cell[isInEdge, 1], weights=-val[isInEdge], minlength=NC)
        a /=2
        return a

    def cell_area(self, index=None):
        #TODO: 3D Case
        NC = self.number_of_cells()
        node = self.node
        edge = self.ds.edge
        edge2cell = self.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        w = np.array([[0, -1], [1, 0]], dtype=self.itype)
        v= (node[edge[:, 1], :] - node[edge[:, 0], :])@w
        val = np.sum(v*node[edge[:, 0], :], axis=1)
        a = np.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a+= np.bincount(edge2cell[isInEdge, 1], weights=-val[isInEdge], minlength=NC)
        a /=2
        return a

    def bc_to_point(self, bcs, index=None):
        node = self.entity('node')
        edge = self.entity('edge')
        index = index if index is not None else np.s_[:]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    def edge_bc_to_point(self, bcs, index=None):
        node = self.entity('node')
        edge = self.entity('edge')
        index = index if index is not None else np.s_[:]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    def tri_refine(self):
        """
        add barycenter and connect them the vertices
        """
        NN = self.number_of_nodes()
        bc = self.entity_barycenter('cell')
        node = self.entity('node')
        edge = self.entity('edge')
        cell2edge = self.ds.cell_to_edge()

        NV = self.number_of_vertices_of_cells()
        NC = len(cell2edge)
        node = np.r_['0', self.node, bc]
        cell = np.zeros((NC, 3), dtype=self.itype)
        cell[:, 0] = np.repeat(range(NN, NN + len(bc)), NV)
        cell[:, 1:] = edge[cell2edge]
        return node, cell

    def refine(self, isMarkedCell):

        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        NV = self.number_of_vertices_of_cells()

        NC1 = isMarkedCell.sum()
        if  NC1 > 0:
            NC0 = NC - NC1
            cell, cellLocation = self.entity('cell')
            node = self.entity('node')
            edge = self.entity('edge')

            isMarkedEdge = np.zeros(NE, dtype=np.bool_)
            edge2cell = self.ds.edge_to_cell()
            cell2edge = self.ds.cell_to_edge(return_sparse=False)
            isMarkedEdge[isMarkedCell[edge2cell[:, 0]]] = True
            isMarkedEdge[isMarkedCell[edge2cell[:, 1]]] = True

            NV0 = NV[~isMarkedCell]
            NV1 = NV[isMarkedCell]

            s0 = np.sum(NV0)
            s1 = np.sum(NV1)

            newCellLocation = np.zeros(NC0 + s1 + 1, dtype=self.itype)
            newCellLocation[:NC0] = cellLocation[0:-1][~isMarkedCell]
            newCellLocation[NC0:-1] = np.arange(s0, s0+4*s1, 4)
            newCellLocation[-1] = s0+4*s1

            emap = np.zeros(NE, dtype=self.itype)
            NE1 = isMarkedEdge.sum()
            emap[isMarkedEdge] = range(NE1)
            cmap = np.zeros(NC, dtype=self.itype)
            cmap[isMarkedCell] = range(NC1)

            newCell = np.zeros(s0+4*s1, dtype=self.itype)
            flag0 = np.repeat(isMarkedCell, NV)
            newCell[:s0] = cell[~flag0]
            newCell[s0::4] = np.repeat(range(NN+NE1, NN+NE1+NC1), NV1)
            newCell[s0+1::4] = cell2edge[cellLocation[isMarkedCell]]
            newCell[s0+2::4] = cell2edge[cellLocation[isMarkedCell]]
            newCell[s0+3::4] = cell2edge[cellLocation[isMarkedCell]]

            ec = self.entity_barycenter('edge')
            cc = self.entity_barycenter('cell')

    def print(self):
        print("Node:\n", self.node)
        print("Cell:\n", self.ds.cell)
        print("Edge:\n", self.ds.edge)
        print("Edge2cell:\n", self.ds.edge2cell)
        print("Cell2edge:\n", self.ds.cell_to_edge(return_sparse=False))
        print("edge norm:\n", self.edge_unit_normal())
        print("cell barycenter:\n", self.entity_barycenter('cell'))



class PolygonMeshDataStructure():
    def __init__(self, NN, cell, cellLocation, topdata=None):
        self.NN = NN
        self.NC = cellLocation.shape[0] - 1

        self.cell = cell
        self.cellLocation = cellLocation

        self.itype = cell.dtype

        if topdata is None:
            self.construct()
        else:
            self.edge = topdata[0]
            self.edge2cell = topdata[1]
            self.NE = len(edge)


    def reinit(self, NN, cell, cellLocation):
        self.NN = NN
        self.NC = cellLocation.shape[0] - 1

        self.cell = cell
        self.cellLocation = cellLocation
        self.construct()

    def clear(self):
        self.edge = None
        self.edge2cell = None

    def number_of_vertices_of_cells(self):
        cellLocation = self.cellLocation
        return cellLocation[1:] - cellLocation[0:-1]

    def number_of_nodes_of_cells(self):
        cellLocation = self.cellLocation
        return cellLocation[1:] - cellLocation[0:-1]

    def number_of_edges_of_cells(self):
        cellLocation = self.cellLocation
        return cellLocation[1:] - cellLocation[0:-1]

    def total_edge(self):
        cell = self.cell
        cellLocation = self.cellLocation

        NN = self.NN
        NC = self.NC
        NV = self.number_of_nodes_of_cells()

        totalEdge = np.zeros((cell.shape[0], 2), dtype=self.itype)
        totalEdge[:, 0] = cell
        totalEdge[:-1, 1] = cell[1:]
        totalEdge[cellLocation[1:] - 1, 1] = cell[cellLocation[:-1]]
        return totalEdge

    def construct(self):
        NC = self.NC

        cell = self.cell
        cellLocation = self.cellLocation
        NV = self.number_of_vertices_of_cells()

        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.NE = NE
        self.edge2cell = np.zeros((NE, 4), dtype=self.itype)

        i1 = np.zeros(NE, dtype=self.itype)
        i1[j] = np.arange(len(cell))

        self.edge = totalEdge[i0]

        cellIdx = np.repeat(range(NC), NV)
 
        localIdx = ranges(NV)

        self.edge2cell[:, 0] = cellIdx[i0]
        self.edge2cell[:, 1] = cellIdx[i1]
        self.edge2cell[:, 2] = localIdx[i0]
        self.edge2cell[:, 3] = localIdx[i1]
        self.cell2edge = j

    def cell_to_node(self):
        NN = self.NN
        NC = self.NC
        NE = self.NE

        cell = self.cell
        cellLocation = self.cellLocation
        NV = self.number_of_vertices_of_cells()
        I = np.repeat(range(NC), NV)
        J = cell

        val = np.ones(len(cell), dtype=np.bool_)
        cell2node = csr_matrix((val, (I, J)), shape=(NC, NN), dtype=np.bool_)
        return cell2node

    def cell_to_edge(self, return_sparse=False):
        NE = self.NE
        NC = self.NC

        edge2cell = self.edge2cell
        cell = self.cell
        cellLocation = self.cellLocation

        if return_sparse:
            J = np.arange(NE)
            val = np.ones((NE,), dtype=np.bool_)
            cell2edge = coo_matrix((val, (edge2cell[:,0], J)), shape=(NC, NE), dtype=np.bool_)
            cell2edge += coo_matrix((val, (edge2cell[:,1], J)), shape=(NC, NE), dtype=np.bool_)
            return cell2edge.tocsr()
        else:
            cell2edge = np.zeros(cell.shape[0], dtype=self.itype)
            cell2edge[cellLocation[edge2cell[:, 0]] + edge2cell[:, 2]] = range(NE)
            cell2edge[cellLocation[edge2cell[:, 1]] + edge2cell[:, 3]] = range(NE)
            return cell2edge

    def cell_to_edge_sign(self, return_sparse=True):
        NE = self.NE
        NC = self.NC
        edge2cell = self.edge2cell
        cell = self.cell
        cellLocation = self.cellLocation
        if return_sparse:
            val = np.ones((NE,), dtype=np.bool_)
            cell2edgeSign = csr_matrix((val, (edge2cell[:,0], range(NE))), shape=(NC,NE), dtype=np.bool_)
            return cell2edgeSign
        else:
            cell2edgeSign = np.zeros(cell.shape[0], dtype=self.itype)
            isInEdge = edge2cell[:, 0] != edge2cell[:, 1]
            cell2edgeSign[cellLocation[edge2cell[:, 0]] + edge2cell[:, 2]] = 1
            cell2edgeSign[cellLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]] = -1
            return cell2edgeSign

    def cell_to_cell(self):
        NC = self.NC
        edge2cell = self.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        val = np.ones(isInEdge.sum(), dtype=np.bool_)
        cell2cell = coo_matrix(
                (val, (edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])),
                shape=(NC,NC), dtype=np.bool_)
        cell2cell+= coo_matrix(
                (val, (edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])), 
                shape=(NC,NC), dtype=np.bool_)
        return cell2cell.tocsr()

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE

        edge = self.edge
        if return_sparse:
            val = np.ones((NE,), dtype=np.bool_)
            edge2node = coo_matrix((val, (edge[:,0], edge[:,1])), shape=(NE, NN), dtype=np.bool_)
            edge2node+= coo_matrix((val, (edge[:,1], edge[:,0])), shape=(NE, NN), dtype=np.bool_)
            return edge2node.tocsr()
        else:
            return edge

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.transpose()

    def edge_to_cell(self, return_sparse=False):
        NE = self.NE
        NC = self.NC
        edge2cell = self.edge2cell
        if return_sparse:
            val = np.ones(NE, dtype=np.bool_)
            edge2cell = coo_matrix((val, (range(NE), edge2cell[:,0])), shape=(NE, NC), dtype=np.bool_)
            edge2cell+= coo_matrix((val, (range(NE), edge2cell[:,1])), shape=(NE, NC), dtype=np.bool_)
            return edge2cell.tocsr()
        else:
            return edge2cell

    def node_to_node(self):
        NN = self.NN
        edge = self.edge
        return self.node_to_node_in_edge(NN, edge)

    def node_to_node_in_edge(self, NN, edge):
        I = edge.flatten()
        J = edge[:, [1, 0]].flatten()
        val = np.ones(2*edge.shape[0], dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE

        edge = self.edge

        val = np.ones((NE,), dtype=np.bool_)
        node2edge = coo_matrix((val, (edge[:,0], range(NE))), shape=(NE, NN), dtype=np.bool_)
        node2edge+= coo_matrix((val, (edge[:,1], range(NE))), shape=(NE, NN), dtype=np.bool_)
        return node2edge.tocsr()

    def node_to_cell(self):
        NN = self.NN
        NC = self.NC

        cell = self.cell
        NV = self.number_of_vertices_of_cells()
        I = cell
        J = np.repeat(range(NC), NV)
        val = np.ones(cell.shape[0], dtype=np.bool_)
        node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool_)
        return node2cell

    def boundary_edge_to_edge(self):
        NN = self.NN
        edge = self.edge
        index = self.boundary_edge_index()
        bdEdge = edge[index]
        n = bdEdge.shape[0]
        val = np.ones(n, dtype=np.bool_)
        m0 = csr_matrix((val, (range(n), bdEdge[:, 0])), shape=(n, NN), dtype=np.bool_)
        m1 = csr_matrix((val, (range(n), bdEdge[:, 1])), shape=(n, NN), dtype=np.bool_)
        _, nex = (m0*m1.T).nonzero()
        _, pre = (m1*m0.T).nonzero()
        return index[nex], index[pre]


    def boundary_node_flag(self):
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdNode = np.zeros(NN, dtype=np.bool_)
        isBdNode[edge[isBdEdge,:]] = True
        return isBdNode

    def boundary_edge_flag(self):
        edge2cell = self.edge2cell
        return edge2cell[:,0] == edge2cell[:,1]

    def boundary_edge(self):
        edge = self.edge
        return edge[self.boundary_edge_index()]

    def boundary_cell_flag(self):
        NC = self.NC
        edge2cell = self.edge2cell
        isBdEdge = self.boundary_edge_flag()

        isBdCell = np.zeros(NC, dtype=np.bool_)
        isBdCell[edge2cell[isBdEdge,0]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdNode = self.boundary_node_flag()
        idx, = np.nonzero(isBdNode)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx
