
import numpy as np
import torch
from torch import Tensor

from .mesh_ds import MeshDataStructure, Redirector


class Mesh3dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of 3-d mesh.\
           This is an abstract class and can not be used directly.
    """
    # Variables
    face2cell: Tensor
    cell2edge: Tensor

    # Constants
    TD = 3
    localFace2edge: Tensor
    localEdge2face: Tensor

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
        return self.cell

    def cell_to_edge(self):
        return self.cell2edge

    def cell_to_edge_sign(self):
        pass

    def cell_to_face(self):
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        face2cell = self.face2cell
        NFC = self.number_of_faces_of_cells()
        cell2face = torch.zeros((NC, NFC), dtype=self.itype)
        cell2face[face2cell[:, 0], face2cell[:, 2]] = range(NF)
        cell2face[face2cell[:, 1], face2cell[:, 3]] = range(NF)
        return cell2face

    def cell_to_face_sign(self):
        pass

    def cell_to_cell(self):
        pass


    ### Face ###

    def face_to_node(self):
        return self.face

    def face_to_edge(self):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        face2edge = cell2edge[
                face2cell[:, [0]],
                localFace2edge[face2cell[:, 2]]
                ]
        return face2edge

    def face_to_face(self):
        pass

    def face_to_cell(self):
        return self.face2cell


    ### Edge ###

    def edge_to_node(self):
        return self.edge

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
        NN = self.number_of_nodes()
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = torch.zeros((NN,), dtype=torch.bool)
        isBdPoint[face[isBdFace, :]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        NE = self.number_of_edges()
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = torch.zeros((NE,), dtype=torch.bool)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge

    def boundary_face_flag(self):
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.number_of_cells()
        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = torch.zeros((NC,), dtype=torch.bool)
        isBdCell[face2cell[isBdFace, 0]] = True
        return isBdCell
