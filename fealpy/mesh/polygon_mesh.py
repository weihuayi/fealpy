import numpy as np
from typing import Union
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, coo_matrix

from ..common import ranges
from ..quadrature import TriangleQuadrature
from ..quadrature import GaussLegendreQuadrature

from .mesh_base import Mesh2d
from .mesh_data_structure import Mesh2dDataStructure

class PolygonMesh(Mesh2d):
    def __init__(self, node: NDArray, cell: NDArray, cellLocation=None, topdata=None):
        self.node = node
        if cellLocation is None:
            if len(cell.shape)  == 2:
                NC = cell.shape[0]
                NV = cell.shape[1]
                cell = cell.reshape(-1)
                cellLocation = np.arange(0, (NC+1)*NV, NV)
            else:
                raise ValueError("Miss `cellLocation` array!")

        self.ds = PolygonMeshDataStructure(node.shape[0], cell, cellLocation,
                topdata=None)
        self.meshtype = 'polygon'
        self.itype = cell.dtype
        self.ftype = node.dtype

        self.cell_data = {}
        self.node_data = {}
        self.edge_data = {}
        self.face_data = self.edge_data
        self.mesh_data = {}

    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(q)

    def entity_barycenter(self, etype: Union[int, str]='cell', index=np.s_[:]):

        node = self.entity('node')
        GD = self.geo_dimension()

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

    def edge_bc_to_point(self, bcs: NDArray, index=np.s_[:]):
        """
        @brief 给出边上的重心坐标，返回其对应的插值点
        """
        node = self.entity('node')
        edge = self.entity('edge')
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    bc_to_point = edge_bc_to_point
    face_bc_to_point = edge_bc_to_point

    def cell_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        raise NotImplementedError

    def edge_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        raise NotImplementedError

    face_to_ipoint = edge_to_ipoint

    def shape_function(self, bc: NDArray, p: int) -> NDArray:
        raise NotImplementedError

    def grad_shape_function(self, bc: NDArray, p: int, index=np.s_[:]) -> NDArray:
        raise NotImplementedError

    def interpolation_points(self):
        raise NotImplementedError

    def multi_index_matrix(self):
        raise NotImplementedError

    def node_to_ipoint(self):
        raise NotImplementedError

    def number_of_global_ipoints(self, p: int) -> int:
        raise NotImplementedError

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        raise NotImplementedError

    def uniform_refine(self, n: int=1) -> None:
        raise NotImplementedError

    @classmethod
    def from_mesh(cls, mesh: Mesh2d):
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


class PolygonMeshDataStructure(Mesh2dDataStructure):
    def __init__(self, NN: int, cell: NDArray, cellLocation: NDArray, topdata=None):
        self.TD = 2
        self.NN = NN
        self.NC = cellLocation.shape[0] - 1

        self._cell = cell
        self.cellLocation = cellLocation

        self.itype = cell.dtype

        if topdata is None:
            self.construct()
        else:
            self.edge = topdata[0]
            self.edge2cell = topdata[1]
            self.NE = len(self.edge)
            self.NF = self.NE


    def reinit(self, NN: int, cell: NDArray, cellLocation: NDArray):
        self.NN = NN
        self.NC = cellLocation.shape[0] - 1

        self._cell = cell
        self.itype = cell.dtype
        self.cellLocation = cellLocation
        self.construct()

    def number_of_vertices_of_cells(self):
        return self.cellLocation[1:] - self.cellLocation[0:-1]

    number_of_edges_of_cells = number_of_vertices_of_cells
    number_of_faces_of_cells = number_of_vertices_of_cells

    def total_edge(self) -> NDArray:
        totalEdge = np.zeros((self._cell.shape[0], 2), dtype=self.itype)
        totalEdge[:, 0] = self._cell
        totalEdge[:-1, 1] = self._cell[1:]
        totalEdge[self.cellLocation[1:] - 1, 1] = self._cell[self.cellLocation[:-1]]
        return totalEdge

    total_face = total_edge

    def construct(self):
        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.NE = NE
        self.NF = NE
        self.edge2cell = np.zeros((NE, 4), dtype=self.itype)

        i1 = np.zeros(NE, dtype=self.itype)
        i1[j] = np.arange(len(self._cell))

        self.edge = totalEdge[i0]

        NV = self.number_of_vertices_of_cells()
        cellIdx = np.repeat(range(self.NC), NV)

        localIdx = ranges(NV)

        self.edge2cell[:, 0] = cellIdx[i0]
        self.edge2cell[:, 1] = cellIdx[i1]
        self.edge2cell[:, 2] = localIdx[i0]
        self.edge2cell[:, 3] = localIdx[i1]

    @property
    def cell(self):
        return np.hsplit(self._cell, self.cellLocation[1:-1])

    def cell_to_node(self):
        NN = self.NN
        NC = self.NC
        NE = self.NE

        NV = self.number_of_vertices_of_cells()
        I = np.repeat(range(NC), NV)
        J = cell

        val = np.ones(len(self._cell), dtype=np.bool_)
        cell2node = csr_matrix((val, (I, J)), shape=(NC, NN), dtype=np.bool_)
        return cell2node

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

    face_to_cell = edge_to_cell

    ### boundary ###

    def boundary_edge_flag(self):
        return self.boundary_face_flag()

    def boundary_edge(self):
        edge = self.edge
        return edge[self.boundary_edge_index()]
