
import numpy as np
import torch
from torch import Tensor

from .mesh_ds import MeshDataStructure, Redirector


class Mesh2dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 2-d mesh.\
           This is an abstract class and can not be used directly.
    """
    #Variables
    face: Redirector[Tensor] = Redirector('edge')
    edge2cell: Tensor

    # Constants
    TD = 2
    NVF: Redirector[int] = Redirector('NVE')
    NFC: Redirector[int] = Redirector('NEC')

    def total_edge(self) -> Tensor:
        """
        @brief Return total edges in mesh.

        @return: Tensor with shape (NC*NEC, 2) where NN is number of cells,\
                 NEC is number of edges in each cell.

        @note: There are 2 nodes constructing an edge.
        """
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, 2)
        return totalEdge

    total_face = total_edge

    def construct(self):
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()

        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
                return_index=True,
                return_inverse=True,
                axis=0)

        self.edge = totalEdge[i0, :]

        NE = i0.shape[0]
        self.edge2cell = torch.zeros((NE, 4), dtype=self.itype, device=self.device)

        i0 = torch.from_numpy(i0)
        i1 = torch.zeros(NE, dtype=self.itype)
        i1[j] = torch.arange(NEC*NC, dtype=self.itype)

        self.edge2cell[:, 0] = i0 // NEC
        self.edge2cell[:, 1] = i1 // NEC
        self.edge2cell[:, 2] = i0 % NEC
        self.edge2cell[:, 3] = i1 % NEC

    def clear(self):
        self.edge = None
        self.edge2cell = None


    ### Cell ###

    def cell_to_node(self):
        """
        @brief Neighber info from cell to node.

        @return: A tensor with shape (NC, NVC), containing indexes of nodes in\
                 every cells.
        """
        return self.cell

    def cell_to_edge(self):
        """
        @brief Neighber info from cell to edge.

        @return: A tensor with shape (NC, NEC), containing indexes of edges in\
                 every cells.
        """
        NC = self.number_of_cells()
        NE = self.number_of_edges()
        NEC = self.number_of_edges_of_cells()
        edge2cell = self.edge2cell

        cell2edge = torch.zeros((NC, NEC), dtype=self.itype, device=self.device)
        cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = torch.arange(NE)
        cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = torch.arange(NE)
        return cell2edge

    def cell_to_edge_sign(self):
        NC = self.number_of_cells()
        NEC = self.NEC

        edge2cell = self.edge2cell

        cell2edgeSign = torch.zeros((NC, NEC), dtype=torch.bool, device=self.device)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True

        return cell2edgeSign

    cell_to_face = cell_to_edge
    cell_to_face_sign = cell_to_edge_sign

    def cell_to_cell(self):
        pass


    ### Edge ###

    def edge_to_node(self):
        """
        @brief Neighber info from edge to node.

        @return: A tensor with shape (NE, NEC), containing indexes of nodes in\
                 every edges.
        """
        return self.edge

    def edge_to_edge(self):
        pass

    edge_to_face = edge_to_edge

    def edge_to_cell(self):
        """
        @brief Neighber info from edge to cell.

        @return: A tensor with shape (NE, 4), providing 4 features for each edge:
        - (0) Index of cell in the left of edge;
        - (1) Index of cell in the right of edge;
        - (2) Local index of the edge in the left cell;
        - (3) Locel index of the edge in the right cell.
        """
        return self.edge2cell


    ### Face ###

    face_to_cell = edge_to_cell
    face_to_face = edge_to_edge
    face_to_edge = edge_to_edge
    face_to_node = edge_to_node


    ### Node ###

    def node_to_node(self):
        pass

    def node_to_edge(self):
        pass

    node_to_face = node_to_edge

    def node_to_cell(self):
        pass


    ### Boundary ###

    def boundary_node_flag(self):
        """
        @brief Boundary node indicator.

        @return: A bool tensor with shape (NN, ) to indicate if a node is\
                 on the boundary or not.
        """
        NN = self.number_of_nodes()
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdNode = torch.zeros((NN,), dtype=torch.bool, device=self.device)
        isBdNode[edge[isBdEdge, :]] = True
        return isBdNode

    def boundary_edge_flag(self):
        """
        @brief Boundary edge indicator.

        @return: A bool tensor with shape (NE, ) to indicate if an edge is\
                 part of boundary or not.
        """
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    boundary_face_flag = boundary_edge_flag

    def boundary_cell_flag(self):
        """
        @brief Boundary cell indicator.

        @return: A bool tensor with shape (NC, ) to indicate if a cell locats\
                 next to the boundary.
        """
        NC = self.number_of_cells()
        edge2cell = self.edge2cell
        isBdCell = torch.zeros((NC,), dtype=torch.bool, device=self.device)
        isBdEdge = self.boundary_edge_flag()
        isBdCell[edge2cell[isBdEdge, 0]] = True
        return isBdCell
