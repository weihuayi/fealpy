
import numpy as np
import torch
from torch import Tensor

from .mesh_ds import MeshDataStructure, Redirector


class Mesh3dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 3-d mesh.\
           This is an abstract class and can not be used directly.
    """
    TD = 3
    face2cell: Tensor
    cell2edge: Tensor

    def total_edge(self):
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, localEdge.shape[1])
        return totalEdge

    def total_face(self):
        cell = self.cell
        localFace = self.localFace
        totalFace = cell[:, localFace].reshape(-1, localFace.shape[1])
        return totalFace

    def construct(self):
        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()

        totalFace = self.total_face()
        _, i0, j = np.unique(
                np.sort(totalFace, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)

        self.face = totalFace[i0]

        NF = i0.shape[0]
        self.face2cell = torch.zeros((NF, 4), dtype=self.itype, device=self.device)

        i0 = torch.from_numpy(i0)
        i1 = torch.zeros(NF, dtype=self.itype)
        i1[j] = torch.arange(NFC*NC, dtype=self.itype)

        self.face2cell[:, 0] = i0 // NFC
        self.face2cell[:, 1] = i1 // NFC
        self.face2cell[:, 2] = i0 % NFC
        self.face2cell[:, 3] = i1 % NFC

        totalEdge = self.total_edge()
        self.edge, i2, j = np.unique(
                np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NEC = self.number_of_edges_of_cells()
        self.cell2edge = np.reshape(j, (NC, NEC))

    def clear(self):
        self.face = None
        self.edge = None
        self.face2cell = None
        self.cell2edge = None


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
        return self.cell2edge

    def cell_to_edge_sign(self):
        pass

    def cell_to_face(self):
        pass

    def cell_to_face_sign(self):
        pass

    def cell_to_cell(self):
        pass


    ### Face ###

    def face_to_node(self):
        pass

    def face_to_edge(self):
        pass

    def face_to_face(self):
        pass

    def face_to_cell(self):
        pass


    ### Edge ###

    def edge_to_node(self):
        pass

    def edge_to_edge(self):
        pass

    def edge_to_face(self):
        pass

    def edge_to_cell(self):
        pass


    ### Node ###

    def node_to_node(self):
        pass

    def node_to_edge(self):
        pass

    def node_to_face(self):
        pass

    def node_to_cell(self):
        pass


    ### Boundary ###

    def boundary_node_flag(self):
        pass

    def boundary_edge_flag(self):
        pass

    def boundary_face_flag(self):
        pass

    def boundary_cell_flag(self):
        pass
