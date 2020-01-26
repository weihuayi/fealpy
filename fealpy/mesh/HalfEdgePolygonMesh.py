import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from ..common import ranges
from .mesh_tools import unique_row, find_entity, show_mesh_2d
from ..quadrature import TriangleQuadrature
from .Mesh2d import Mesh2d

class HalfEdgePolygonMesh(Mesh2d):
    def __init__(self, node, halfedge, NC):
        """

        Parameters
        ----------
        node : (NN, GD)
        halfedge : (2*NE, 5),
        """
        self.node = node
        self.ds = HalfEdgePolygonMeshDataStructure(node.shape[0], NC, halfedge)
        self.meshtype = 'hepolygon'
        self.itype = halfedge.dtype
        self.ftype = node.dtype

    @classmethod
    def from_polygonmesh(cls, mesh):
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NV = mesh.number_of_vertices_of_cells()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell, cellLocation = mesh.entity('cell')
        cell2edge = mesh.ds.cell_to_edge(sparse=False)
        edge2cell = mesh.ds.edge_to_cell()
        cell2edgeSign = mesh.ds.cell_to_edge_sign(sparse=False)
        cell2edgeSign[cell2edgeSign==1] = 0
        cell2edgeSign[cell2edgeSign==-1] = NE

        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        nex, pre = mesh.ds.boundary_edge_to_edge()

        halfedge = np.zeros((2*NE, 5), dtype=mesh.itype)
        # 指向的顶点
        halfedge[:NE, 0] = edge[:, 1]
        halfedge[NE:, 0] = edge[:, 0]

        # 指向的单元
        halfedge[:NE, 1] = edge2cell[:, 0]
        halfedge[NE:, 1] = edge2cell[:, 1]
        halfedge[NE:, 1][~isInEdge] = NC

        # 在指向单元中的下一条边
        idx = cellLocation[edge2cell[:, 0]] + (edge2cell[:, 2] + 1)%NV[edge2cell[:,  0]]
        halfedge[:NE, 2] = cell2edge[idx] + cell2edgeSign[idx]

        idx = cellLocation[edge2cell[isInEdge, 1]] + (edge2cell[isInEdge, 3] + 1)%NV[edge2cell[isInEdge,  1]]
        halfedge[NE:, 2][isInEdge] = cell2edge[idx] + cell2edgeSign[idx]
        halfedge[NE:, 2][~isInEdge] = NE + nex

        # 在指向单元中的上一条边
        idx = cellLocation[edge2cell[:, 0]] + (edge2cell[:, 2] - 1)%NV[edge2cell[:,  0]]
        halfedge[:NE, 3] = cell2edge[idx] + cell2edgeSign[idx]

        idx = cellLocation[edge2cell[isInEdge, 1]] + (edge2cell[isInEdge, 3] - 1)%NV[edge2cell[isInEdge,  1]]
        halfedge[NE:, 3][isInEdge] = cell2edge[idx] + cell2edgeSign[idx]
        halfedge[NE:, 3][~isInEdge] = NE + pre

        # 相反的halfedge
        halfedge[:NE, 4] = range(NE, 2*NE)
        halfedge[NE:, 4] = range(NE)
        return cls(node, halfedge, NC)

    def entity(self, etype=2):
        if etype in {'cell', 2}:
            return self.ds.cell_to_node(sparse=False)
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge_to_node(sparse=False)
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
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge_to_node(sparse=False)
            bc = np.sum(node[edge, :], axis=1).reshape(-1, dim)/edge.shape[1]
        elif etype in {'node', 1}:
            bc = node
        return bc

    def cell_area(self, index=None):
        NC = self.number_of_cells()
        node = self.entity('node')
        halfedge = self.ds.halfedge
        isInHEdge = halfedge[:, 1] != -1

        e0 = halfedge[halfedge[isInHEdge, 3], 0]
        e1 = halfedge[isInHEdge, 0]

        w = np.array([[0, -1], [1, 0]], dtype=np.int)
        v= (node[e1] - node[e0])@w
        val = np.sum(v*node[e0], axis=1)

        a = np.zeros(NC+1, dtype=self.ftype)
        np.add.at(a, halfedge[isInHEdge, 1], val)
        a /=2
        return a[:-1]

    def edge_bc_to_point(self, bcs, index=None):
        node = self.entity('node')
        edge = self.entity('edge')
        index = index if index is not None else np.s_[:]
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    def refine(self, isMarkedCell):
        isMarkedCell = np.r_['0', isMarkedCell, False]
        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        NV = self.number_of_vertices_of_cells()

        halfedge = self.ds.halfedge
        isInHEdge = (halfedge[:, 1] != NC)

        # 标记边
        isMarkedHEdge = isMarkedCell[halfedge[:, 1]]
        flag = ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]
        isMarkedHEdge[flag] = True

        # 细分边
        node = self.entity('node')
        isMarkedHEdge0 = isMarkedHEdge[:NE]
        halfedge0 = halfedge[:NE]
        halfedge1 = halfedge[NE:]
        ec = (node[halfedge0[isMarkedHEdge0, 0]] + node[halfedge1[isMarkedHEdge0, 0]])/2
        NE1 = len(ec)

        halfedge = np.zeros((2*(NE + NE1), 5), dtype=self.itype)
        idx, = np.nonzero(isMarkedHEdge0)
        halfedge[NE:NE+NE1, 0] = range(NN, NN+NE1)
        halfedge[NE:NE+NE1, 1] = halfedge0[isMarkedHEdge0, 1]
        halfedge[NE:NE+NE1, 2] = idx
        halfedge[NE:NE+NE1, 3] = halfedge0[isMarkedHEdge0, 3]
        flag = halfedge0[isMarkedHEdge0, 3] >= NE
        halfedge[NE:NE+NE1, 3][flag] += NE1
        halfedge[NE:NE+NE1, 4] = NE + NE1 + idx

        halfedge[2*NE+NE1:, 0] = range(NN, NN+NE1)
        halfedge[2*NE+NE1:, 1] = halfedge1[isMarkedHEdge0, 1]
        halfedge[2*NE+NE1:, 2] = NE1 + NE + idx
        flag = halfedge1[isMarkedHEdge0, 3] >= NE
        halfedge[2*NE+NE1:, 3] = halfedge1[isMarkedHEdge0, 3]
        halfedge[2*NE+NE1:, 3][flag] += NE1
        halfedge[2*NE+NE1:, 4] = idx

        halfedge[:NE, 0] = halfedge0[:, 0]
        halfedge[:NE, 1] = halfedge0[:, 1]
        halfedge[:NE, 3] = halfedge0[:, 3]
        flag = halfedge0[:, 3] >= NE
        halfedge[:NE, 3][flag] += NE1
        halfedge[:NE, 3][isMarkedHEdge0] = range(NE, NE+NE1)
        halfedge[:NE, 4] = halfedge0[:, 4]
        flag = halfedge0[:, 4] >= NE
        halfedge[:NE, 4][flag] += NE1
        halfedge[:NE, 4][isMarkedHEdge0] = range(2*NE+NE1, 2*NE+2*NE1)


        halfedge[NE+NE1:2*NE+NE1, 0] = halfedge1[:, 0]
        halfedge[NE+NE1:2*NE+NE1, 1] = halfedge1[:, 1]
        halfedge[NE+NE1:2*NE+NE1, 3] = halfedge1[:, 3]
        flag = halfedge1[:, 3] >= NE
        halfedge[NE+NE1:2*NE+NE1, 3][flag] += NE1
        halfedge[NE+NE1:2*NE+NE1, 3][isMarkedHEdge0] = range(2*NE+NE1,
                2*NE+2*NE1)
        halfedge[NE+NE1:2*NE+NE1, 4] = halfedge1[:, 4]
        flag = halfedge1[:, 4] >= NE
        halfedge[NE+NE1:2*NE+NE1, 4][flag] += NE1
        halfedge[NE+NE1:2*NE+NE1, 4][isMarkedHEdge0] = range(NE, NE+NE1)


        halfedge[halfedge[:, 3], 2] = range(2*NE+2*NE1)
        self.node = np.r_['0', node, ec]
        self.ds.reinit(NN+NE1, NC, halfedge)



        # 细分单元


    def print(self):
        cell, cellLocation = self.entity('cell')
        print("cell:\n", cell)
        print("cellLocation:\n", cellLocation)
        print("cell2edge:\n", self.ds.cell_to_edge(sparse=False))

        print("edge:")
        for i, val in enumerate(self.entity('edge')):
            print(i, ":", val)
        print("halfedge:")
        for i, val in enumerate(self.ds.halfedge):
            print(i, ":", val)

class HalfEdgePolygonMeshDataStructure():
    def __init__(self, NN, NC, halfedge):
        self.NN = NN
        self.NC = NC
        self.NE = len(halfedge)//2
        self.NF = self.NE
        self.halfedge = halfedge
        self.itype = halfedge.dtype

        self.cell2hedge = np.zeros(NC+1, dtype=self.itype)
        flag = halfedge[:, 1] != -1
        idx = np.arange(2*self.NE)
        self.cell2hedge[halfedge[flag, 1]] = idx[flag]

    def reinit(self, NN, NC, halfedge):
        self.NN = NN
        self.NC = NC
        self.NE = len(halfedge)//2
        self.NF = self.NE
        self.halfedge = halfedge
        self.itype = halfedge.dtype

        self.cell2hedge = np.zeros(NC+1, dtype=self.itype)
        flag = halfedge[:, 1] != -1
        idx = np.arange(2*self.NE)
        self.cell2hedge[halfedge[flag, 1]] = idx[flag]

    def number_of_vertices_of_cells(self):
        NC = self.NC
        halfedge = self.halfedge
        NV = np.zeros(NC+1, dtype=self.itype)
        np.add.at(NV, halfedge[:, 1], 1)
        return NV[:NC]

    def number_of_nodes_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.number_of_vertices_of_cells()

    def number_of_face_of_cells(self):
        return self.number_of_vertices_of_cells()

    def cell_to_node(self, sparse=True):
        NN = self.NN
        NC = self.NC
        NE = self.NE

        halfedge = self.halfedge
        isInHEdge = (halfedge[:, 1] != NC)

        if sparse:
            val = np.ones(isInHEdge.sum(), dtype=np.bool)
            I = halfedge[isInHEdge, 1]
            J = halfedge[isInHEdge, 0]
            cell2node = csr_matrix((val, (I.flat, J.flat)), shape=(NC, NN), dtype=np.bool)
            return cell2node
        else:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2node = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.cell2hedge.copy()[:NC]
            idx = cellLocation[:-1].copy()
            cell2node[idx] = halfedge[current, 0]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while isNotOK.sum() > 0:
               current[isNotOK] = halfedge[current[isNotOK], 2]
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
               isNotOK = (NV0 < NV)
            return cell2node, cellLocation

    def cell_to_edge(self, sparse=True):
        NE = self.NE
        NC = self.NC

        halfedge = self.halfedge

        if sparse:
            J = np.arange(NE)
            isInHEdge = halfedge[NE:, 1] != NC 
            val = np.ones((NE,), dtype=np.bool)
            cell2edge = coo_matrix((val, (halfedge[:NE, 1], J)), shape=(NC, NE), dtype=np.bool)
            cell2edge+= coo_matrix((val[isInEdge], (halfedge[NE:, 1][isInEdge],
                J[isInHEdge])), shape=(NC, NE), dtype=np.bool)
            return cell2edge.tocsr()
        else:
            NV = self.number_of_vertices_of_cells()
            cellLocation = np.zeros(NC+1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2edge = np.zeros(cellLocation[-1], dtype=self.itype)
            current = halfedge[self.cell2hedge[:-1], 2] # 下一个边
            idx = cellLocation[:-1].copy()
            cell2edge[idx] = current%NE
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while isNotOK.sum() > 0:
               current[isNotOK] = halfedge[current[isNotOK], 2]
               idx[isNotOK] += 1
               NV0[isNotOK] += 1
               cell2edge[idx[isNotOK]] = current[isNotOK]%NE
               isNotOK = (NV0 < NV)
            return cell2edge

    def cell_to_face(self):
        return self.cell_to_edge()

    def cell_to_cell(self):
        NC = self.NC
        halfedge = self.halfedge
        isInEdge = (halfedge[:, 1] != NC)
        val = np.ones(isInEdge.sum(), dtype=np.bool)
        I = halfedge[isInEdge, 1]
        J = halfedge[halfedge[isInEdge, 4], 1]
        cell2cell = coo_matrix((val, (I, J)), shape=(NC, NC), dtype=np.bool)
        cell2cell+= coo_matrix((val, (J, I)), shape=(NC, NC), dtype=np.bool)
        return cell2cell.tocsr()

    def edge_to_node(self, sparse=False):
        NN = self.NN
        NE = self.NE

        halfedge = self.halfedge
        if sparse == False:
            edge = np.zeros((NE, 2), dtype=self.itype)
            edge[:, 0] = halfedge[NE:, 0]
            edge[:, 1] = halfedge[halfedge[NE:, 4], 0]
            return edge
        else:
            val = np.ones((NE,), dtype=np.bool)
            edge2node = coo_matrix((val, (range(NE), halfedge[NE:,0])), shape=(NE, NN), dtype=np.bool)
            edge2node+= coo_matrix((val, (range(NE), halfedge[:NE,0])), shape=(NE, NN), dtype=np.bool)
            return edge2node.tocsr()

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.tranpose()

    def edge_to_cell(self, sparse=False):
        NE = self.NE
        NC = self.NC
        edge = self.edge
        halfedge = self.halfedge
        cell2edge = self.cell_to_edge()
        if sparse == False:
            edge2cell = np.zeros((NE, 4), dtype=self.itype)
            return edge2cell
        else:
            val = np.ones(NE, dtype=np.bool)
            edge2cell = coo_matrix((val, (range(NE), edge[:, 2])), shape=(NE, NC), dtype=np.bool)
            edge2cell+= coo_matrix((val, (range(NE), edge[:, 3])), shape=(NE, NC), dtype=np.bool)
            return edge2cell.tocsr()

    def node_to_node(self):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge[:, 0:2].flat
        J = edge[:, 1::-1].flat
        val = np.ones(2*NE, dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool)
        return node2node

    def node_to_cell(self):

        NN = self.NN
        NC = self.NC
        NE = self.NE

        edge = self.edge

        val = np.ones(2*NE, dtype=np.bool)
        I = edge[:, 0:2]
        J = edge[:, [2, 2]]
        node2cell = coo_matrix(
                (val, (I.flat, J.flat)),
                shape=(NN, NC), dtype=np.bool)
        J = halfedge[:, [3, 3]]
        node2cell += coo_matrix(
                (val, (I.flat, J.flat)),
                shape=(NN, NC), dtype=np.bool)
        return node2cell.tocsr()

    def boundary_node_flag(self):
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdNode = np.zeros(NN, dtype=np.bool)
        isBdNode[edge[isBdEdge,:]] = True
        return isBdNode

    def boundary_edge_flag(self):
        NE = self.NE
        edge2cell = self.edge2cell
        return edge2cell[:, 2] == edge2cell[:, 3]

    def boundary_edge(self):
        edge = self.edge
        return edge[self.boundary_edge_index(), 0:2]

    def boundary_cell_flag(self):
        NC = self.NC
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()

        isBdCell = np.zeros(NC, dtype=np.bool)
        isBdCell[edge[isBdEdge, 2]] = True
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
