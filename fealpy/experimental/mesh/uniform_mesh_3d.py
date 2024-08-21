import numpy as np 
from typing import Union, Optional, Sequence, Tuple, Any

from .utils import entitymethod, estr2dim

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S, Union, Tuple
from .. import logger

from .mesh_base import StructuredMesh, TensorMesh
from .plot import Plotable

class UniformMesh3d(StructuredMesh, TensorMesh, Plotable):
    """
    Topological data structure of a structured hexahedral mesh
    The ordering of the nodes in each element is as follows:
      3 ------- 7
     / |       /|
    1 ------- 5 |
    |  |      | |
    |  2------|-6
    | /       |/
    0 ------- 4

    The ordering of the edges in each element is as follows:
          ----- 3---
        / |       / |
       5  |      7  |
      /   9     /   11
      ----1----     |
     |    |    |    |
     |     ----|2---     
     8   /     10  /
     |  4      |  6
     | /       | /
      ----0---- 
    * Edge 0: (0, 4)
    * Edge 1: (1, 5)
    * Edge 2: (2, 6)
    * Edge 3: (3, 7)
    * Edge 4: (0, 2)
    * Edge 5: (1, 3)
    * Edge 6: (4, 6)
    * Edge 7: (5, 7)
    * Edge 8: (0, 1)
    * Edge 9: (2, 3)
    * Edge 10: (4, 5)
    * Edge 11: (6, 7)

    The ordering of the faces in each element is as follows:
          ----------
        / |       / |
       /  | 5    /  |
      /   |    3/   |
      ---------     |
     | 0  |    | 1  |
     |     ----|----     
     |   /2    |   /
     |  /   4  |  /
     | /       | /
      --------- 
    * Face 0: (0, 1, 2, 3)
    * Face 1: (4, 5, 6, 7)
    * Face 2: (0, 1, 4, 5)
    * Face 3: (2, 3, 6, 7)
    * Face 4: (0, 2, 4, 6)
    * Face 5: (1, 3, 5, 7)

    The ordering of entities in the entire mesh is as follows:

    * Node numbering rule: first in the z direction, then in the y direction, and then in the x direction
    * Edge numbering rule: first in the z direction, then in the y direction, and then in the x direction
    * Cell numbering rule: first in the z direction, then in the y direction, and then in the x direction
    """
    def __init__(self, extent=(0, 1, 0, 1, 0, 1), h=(1.0, 1.0, 1.0), 
                origin=(0.0, 0.0, 0.0), itype=None, ftype=None):
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
        super().__init__(TD=3, itype=itype, ftype=ftype)

        # Mesh properties
        self.extent = [int(e) for e in extent]
        self.h = [float(val) for val in h]
        self.origin = [float(o) for o in origin]

        # Mesh dimensions
        self.nx = self.extent[1] - self.extent[0]
        self.ny = self.extent[3] - self.extent[2]
        self.nz = self.extent[5] - self.extent[4]
        self.NN = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.NE = (self.nx + 1) * (self.ny + 1) * self.nz + \
                (self.nx + 1) * self.ny * (self.nz + 1) + \
                self.nx * (self.ny + 1) * (self.nz + 1)
        self.NF = self.nx * self.ny * (self.nz + 1) + \
                self.nx * (self.ny + 1) * self.nz + \
                (self.nx + 1) * self.ny * self.nz
        self.NC = self.nx * self.ny * self.nz

        # Mesh datas
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.celldata = {}
        self.meshdata = {}

        self.meshtype = 'UniformMesh3d'

        # Specify the counterclockwise drawing
        self.ccw = bm.array([0, 2, 3, 1], dtype=self.itype)

        self.cell2edge = self.cell_to_edge()
        self.cell2face = self.cell_to_face()
        # self.face2edge = self.face_to_edge()
        self.face2cell = self.face_to_cell()

        self.localEdge = bm.array([
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7)], dtype=self.itype)
        self.localFace = bm.array([
        (0, 1, 2, 3), (4, 5, 6, 7),  # left and right faces
        (0, 1, 4, 5), (2, 3, 6, 7),  # front and back faces
        (0, 2, 4, 6), (1, 3, 5, 7)], dtype=self.itype)  # bottom and top faces
        self.localFace2edge = bm.array([
        (4, 5, 8, 9), (6, 7, 10, 11),
        (0, 1, 8, 10), (2, 3, 9, 11),
        (0, 2, 4, 6), (1, 3, 5, 7)], dtype=self.itype)


    # 实体生成方法
    @entitymethod(0)
    def _get_node(self) -> TensorLike:
        """
        @brief Generate the nodes in a structured mesh.
        """
        GD = 3
        nx, ny, nz = self.nx, self.ny, self.nz
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1],
               self.origin[2], self.origin[2] + nz * self.h[2]]

        x = bm.linspace(box[0], box[1], nx + 1, dtype=self.ftype)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=self.ftype)
        z = bm.linspace(box[4], box[5], nz + 1, dtype=self.ftype)
        xx, yy, zz = bm.meshgrid(x, y, z, indexing='ij')
        node = bm.concatenate((xx[..., None], yy[..., None], zz[..., None]), axis=-1)

        return node.reshape(-1, GD)
    
    @entitymethod(1)
    def _get_edge(self) -> TensorLike:
        """
        @brief Generate the edges in a structured mesh.
        """
        NN = self.NN
        NE = self.NE
        nx = self.nx
        ny = self.ny
        nz = self.nz

        idx = bm.arange(NN, dtype=self.itype).reshape(nx + 1, ny + 1, nz + 1)
        edge = bm.zeros((NE, 2), dtype=self.itype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            NE0 = 0
            NE1 = nx * (ny + 1) * (nz + 1)
            c = bm.transpose(idx, (0, 1, 2))[:-1, :, :]
            edge[NE0:NE1, 0] = c.flatten()
            edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + (ny + 1) * (nz + 1)

            NE0 = NE1
            NE1 += (nx + 1) * ny * (nz + 1)
            c = bm.transpose(idx, (0, 1, 2))[:, :-1, :]
            edge[NE0:NE1, 0] = c.flatten()
            edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + (nz + 1)

            NE0 = NE1
            NE1 += (nx + 1) * (ny + 1) * nz
            c = bm.transpose(idx, (0, 1, 2))[:, :, :-1]
            edge[NE0:NE1, 0] = c.flatten()
            edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + 1

            return edge
        elif bm.backend_name == 'jax':
            NE0 = 0
            NE1 = nx * (ny + 1) * (nz + 1)
            c = bm.transpose(idx, (0, 1, 2))[:-1, :, :]
            edge = edge.at[NE0:NE1, 0].set(c.flatten())
            edge = edge.at[NE0:NE1, 1].set(edge[NE0:NE1, 0] + (ny + 1) * (nz + 1))

            NE0 = NE1
            NE1 += (nx + 1) * ny * (nz + 1)
            c = bm.transpose(idx, (0, 1, 2))[:, :-1, :]
            edge = edge.at[NE0:NE1, 0].set(c.flatten())
            edge = edge.at[NE0:NE1, 1].set(edge[NE0:NE1, 0] + (nz + 1))

            NE0 = NE1
            NE1 += (nx + 1) * (ny + 1) * nz
            c = bm.transpose(idx, (0, 1, 2))[:, :, :-1]
            edge = edge.at[NE0:NE1, 0].set(c.flatten())
            edge = edge.at[NE0:NE1, 1].set(edge[NE0:NE1, 0] + 1)

            return edge
        else:
            raise NotImplementedError("Backend is not yet implemented.")


    @entitymethod(2)
    def _get_face(self) -> TensorLike:
        """
        @brief Generate the faces in a structured mesh.
        """
        NN = self.NN
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = bm.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        face = bm.zeros((NF, 4), dtype=self.itype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            # TODO 为什么要将 face 转一个方向？
            NF0 = 0
            NF1 = (nx + 1) * ny * nz
            c = idx[:, :-1, :-1]
            face[NF0:NF1, 0] = c.flatten()
            face[NF0:NF1, 1] = face[NF0:NF1, 0] + 1
            face[NF0:NF1, 2] = face[NF0:NF1, 0] + nz + 1
            face[NF0:NF1, 3] = face[NF0:NF1, 2] + 1
            # face[NF0:NF0 + ny * nz, :] = face[NF0:NF0 + ny * nz, [1, 0, 3, 2]]

            NF0 = NF1
            NF1 += nx * (ny + 1) * nz
            c = bm.transpose(idx, (0, 1, 2))[:-1, :, :-1]
            face[NF0:NF1, 0] = c.flatten()
            face[NF0:NF1, 1] = face[NF0:NF1, 0] + 1
            face[NF0:NF1, 2] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
            face[NF0:NF1, 3] = face[NF0:NF1, 2] + 1
            NF2 = NF0 + ny * nz
            N = nz * (ny + 1)
            idx1 = bm.zeros((nx, nz), dtype=self.itype)
            idx1 = bm.arange(NF2, NF2 + nz)
            idx1 = idx1 + bm.arange(0, N * nx, N).reshape(nx, 1)
            idx1 = idx1.flatten()
            # face[idx1] = face[idx1][:, [1, 0, 3, 2]]

            NF0 = NF1
            NF1 += nx * ny * (nz + 1)
            c = bm.transpose(idx, (0, 1, 2))[:-1, :-1, :]
            face[NF0:NF1, 0] = c.flatten()
            face[NF0:NF1, 1] = face[NF0:NF1, 0] + nz + 1
            face[NF0:NF1, 2] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
            face[NF0:NF1, 3] = face[NF0:NF1, 2] + nz + 1
            N = ny * (nz + 1)
            idx2 = bm.zeros((nx, ny), dtype=self.itype)
            idx2 = bm.arange(NF0, NF0 + ny * (nz + 1), nz + 1)
            idx2 = idx2 + bm.arange(0, N * nx, N).reshape(nx, 1)
            idx2 = idx2.flatten()
            # face[idx2] = face[idx2][:, [1, 0, 3, 2]]

            return face
        elif bm.backend_name == 'jax':
            NF0 = 0
            NF1 = (nx + 1) * ny * nz
            c = idx[:, :-1, :-1]
            face = face.at[NF0:NF1, 0].set(c.flatten())
            face = face.at[NF0:NF1, 1].set(face[NF0:NF1, 0] + 1)
            face = face.at[NF0:NF1, 2].set(face[NF0:NF1, 0] + nz + 1)
            face = face.at[NF0:NF1, 3].set(face[NF0:NF1, 2] + 1)
            # face = face.at[NF0:NF0 + ny * nz, :].set(face[NF0:NF0 + ny * nz, [1, 0, 3, 2]])

            NF0 = NF1
            NF1 += nx * (ny + 1) * nz
            c = bm.transpose(idx, (0, 1, 2))[:-1, :, :-1]
            face = face.at[NF0:NF1, 0].set(c.flatten())
            face = face.at[NF0:NF1, 1].set(face[NF0:NF1, 0] + 1)
            face = face.at[NF0:NF1, 2].set(face[NF0:NF1, 0] + (ny + 1) * (nz + 1))
            face = face.at[NF0:NF1, 3].set(face[NF0:NF1, 2] + 1)
            NF2 = NF0 + ny * nz
            N = nz * (ny + 1)
            idx1 = bm.zeros((nx, nz), dtype=self.itype)
            idx1 = bm.arange(NF2, NF2 + nz)
            idx1 = idx1 + bm.arange(0, N * nx, N).reshape(nx, 1)
            idx1 = idx1.flatten()
            # face = face.at[idx1].set(face[idx1][:, [1, 0, 3, 2]])

            NF0 = NF1
            NF1 += nx * ny * (nz + 1)
            c = bm.transpose(idx, (0, 1, 2))[:-1, :-1, :]
            face = face.at[NF0:NF1, 0].set(c.flatten())
            face = face.at[NF0:NF1, 1].set(face[NF0:NF1, 0] + nz + 1)
            face = face.at[NF0:NF1, 2].set(face[NF0:NF1, 0] + (ny + 1) * (nz + 1))
            face = face.at[NF0:NF1, 3].set(face[NF0:NF1, 2] + nz + 1)
            N = ny * (nz + 1)
            idx2 = bm.zeros((nx, ny), dtype=self.itype)
            idx2 = bm.arange(NF0, NF0 + ny * (nz + 1), nz + 1)
            idx2 = idx2 + bm.arange(0, N * nx, N).reshape(nx, 1)
            idx2 = idx2.flatten()
            # face = face.at[idx2].set(face[idx2][:, [1, 0, 3, 2]])

            return face
        else:
            raise NotImplementedError("Backend is not yet implemented.")

    @entitymethod(3)
    def _get_cell(self) -> TensorLike:
        """
        @brief Generate the cells in a structured mesh.
        """
        NN = self.NN
        NC = self.NC
        nx, ny, nz = self.nx, self.ny, self.nz

        idx = bm.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]

        cell = bm.zeros((NC, 8), dtype=self.itype)
        nyz = (ny + 1) * (nz + 1)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            cell[:, 0] = c.flatten()
            cell[:, 1] = cell[:, 0] + 1
            cell[:, 2] = cell[:, 0] + nz + 1
            cell[:, 3] = cell[:, 2] + 1
            cell[:, 4] = cell[:, 0] + nyz
            cell[:, 5] = cell[:, 4] + 1
            cell[:, 6] = cell[:, 2] + nyz
            cell[:, 7] = cell[:, 6] + 1

            return cell
        elif bm.backend_name == 'jax':
            cell = cell.at[:, 0].set(c.flatten())
            cell = cell.at[:, 1].set(cell[:, 0] + 1)
            cell = cell.at[:, 2].set(cell[:, 0] + nz + 1)
            cell = cell.at[:, 3].set(cell[:, 2] + 1)
            cell = cell.at[:, 4].set(cell[:, 0] + nyz)
            cell = cell.at[:, 5].set(cell[:, 4] + 1)
            cell = cell.at[:, 6].set(cell[:, 2] + nyz)
            cell = cell.at[:, 7].set(cell[:, 6] + 1)

            return cell
        else:
            raise NotImplementedError("Backend is not yet implemented.")
    
    
    # 实体拓扑
    def number_of_nodes_of_cells(self):
        return 8

    def number_of_edges_of_cells(self):
        return 12

    def number_of_faces_of_cells(self):
        return 6
    
    def cell_to_edge(self) -> TensorLike:
        """
        @brief 单元和边的邻接关系, 储存每个单元相邻的 12 条边的编号
        """
        NC = self.NC

        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2edge = bm.zeros((NC, 12), dtype=self.itype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            idx0 = bm.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
            cell2edge[:, 0] = idx0[:, :-1, :-1].flatten()
            cell2edge[:, 1] = idx0[:, :-1, 1:].flatten()
            cell2edge[:, 2] = idx0[:, 1:, :-1].flatten()
            cell2edge[:, 3] = idx0[:, 1:, 1:].flatten()

            NE0 = nx * (ny + 1) * (nz + 1)
            idx1 = np.arange((nx + 1) * ny * (nz + 1)).reshape(nx + 1, ny, nz + 1)  
            cell2edge[:, 4] = (NE0 + idx1[:-1, :, :-1]).flatten()
            cell2edge[:, 5] = (NE0 + idx1[:-1, :, 1:]).flatten()
            cell2edge[:, 6] = (NE0 + idx1[1:, :, :-1]).flatten()
            cell2edge[:, 7] = (NE0 + idx1[1:, :, 1:]).flatten()

            NE1 = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1)
            idx2 = np.arange((nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny + 1, nz)
            cell2edge[:, 8] = (NE1 + idx2[:-1, :-1, :]).flatten()
            cell2edge[:, 9] = (NE1 + idx2[:-1, 1:, :]).flatten()
            cell2edge[:, 10] = (NE1 + idx2[1:, :-1, :]).flatten()
            cell2edge[:, 11] = (NE1 + idx2[1:, 1:, :]).flatten()

            return cell2edge
        elif bm.backend_name == 'jax':
            idx0 = bm.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
            cell2edge = cell2edge.at[:, 0].set(idx0[:, :-1, :-1].flatten())
            cell2edge = cell2edge.at[:, 1].set(idx0[:, :-1, 1:].flatten())
            cell2edge = cell2edge.at[:, 2].set(idx0[:, 1:, :-1].flatten())
            cell2edge = cell2edge.at[:, 3].set(idx0[:, 1:, 1:].flatten())

            NE0 = nx * (ny + 1) * (nz + 1)
            idx1 = np.arange((nx + 1) * ny * (nz + 1)).reshape(nx + 1, ny, nz + 1)  
            cell2edge = cell2edge.at[:, 4].set((NE0 + idx1[:-1, :, :-1]).flatten())
            cell2edge = cell2edge.at[:, 5].set((NE0 + idx1[:-1, :, 1:]).flatten())
            cell2edge = cell2edge.at[:, 6].set((NE0 + idx1[1:, :, :-1]).flatten())
            cell2edge = cell2edge.at[:, 7].set((NE0 + idx1[1:, :, 1:]).flatten())

            NE1 = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1)
            idx2 = np.arange((nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny + 1, nz)
            cell2edge = cell2edge.at[:, 8].set((NE1 + idx2[:-1, :-1, :]).flatten())
            cell2edge = cell2edge.at[:, 9].set((NE1 + idx2[:-1, 1:, :]).flatten())
            cell2edge = cell2edge.at[:, 10].set((NE1 + idx2[1:, :-1, :]).flatten())
            cell2edge = cell2edge.at[:, 11].set((NE1 + idx2[1:, 1:, :]).flatten())

            return cell2edge
        else:
            raise NotImplementedError("Backend is not yet implemented.")
        
    def cell_to_face(self):
        """
        @brief 单元和面的邻接关系, 储存每个单元相邻的六个面的编号
        """
        NC = self.NC
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2face = bm.zeros((NC, 6), dtype=self.itype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            # x direction
            idx0 = bm.arange((nx + 1) * ny * nz).reshape(nx + 1, ny, nz)
            cell2face[:, 0] = idx0[:-1, :, :].flatten()
            cell2face[:, 1] = idx0[1:, :, :].flatten()

            # y direction
            NE0 = (nx + 1) * ny * nz
            idx1 = bm.arange(nx * (ny + 1) * nz).reshape(nx, ny + 1, nz)
            cell2face[:, 2] = (NE0 + idx1[:, :-1, :]).flatten()
            cell2face[:, 3] = (NE0 + idx1[:, 1:, :]).flatten()

            # z direction
            NE1 = (nx + 1) * ny * nz + nx * (ny + 1) * nz
            idx2 = bm.arange(nx * ny * (nz + 1)).reshape(nx, ny, nz + 1)
            cell2face[:, 4] = (NE1 + idx2[:, :, :-1]).flatten()
            cell2face[:, 5] = (NE1 + idx2[:, :, 1:]).flatten()

            return cell2face
        elif bm.backend_name == 'jax':
            # x direction
            idx0 = bm.arange((nx + 1) * ny * nz).reshape(nx + 1, ny, nz)
            cell2face = cell2face.at[:, 0].set(idx0[:-1, :, :].flatten())
            cell2face = cell2face.at[:, 1].set(idx0[1:, :, :].flatten())

            # y direction
            NE0 = (nx + 1) * ny * nz
            idx1 = bm.arange(nx * (ny + 1) * nz).reshape(nx, ny + 1, nz)
            cell2face = cell2face.at[:, 2].set((NE0 + idx1[:, :-1, :]).flatten())
            cell2face = cell2face.at[:, 3].set((NE0 + idx1[:, 1:, :]).flatten())

            # z direction
            NE1 = (nx + 1) * ny * nz + nx * (ny + 1) * nz
            idx2 = bm.arange(nx * ny * (nz + 1)).reshape(nx, ny, nz + 1)
            cell2face = cell2face.at[:, 4].set((NE1 + idx2[:, :, :-1]).flatten())
            cell2face = cell2face.at[:, 5].set((NE1 + idx2[:, :, 1:]).flatten())

            return cell2face
        else:
            raise NotImplementedError("Backend is not yet implemented.")
        
    def face_to_edge(self):
        """
        @brief 面和边的邻接关系, 储存每个面相邻的 4 条边的编号
        """

        NE = self.NE
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz
        face2edge = bm.zeros((NF, 4), dtype=self.itype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            # x direction
            NE0 = 0
            NE1 = (nx + 1) * ny * nz
            idx0 = np.arange(nx * (ny + 1) * (nz + 1), NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
            face2edge[NE0:NE1, 0] = idx0[:, :, :-1].flatten()
            face2edge[NE0:NE1, 1] = idx0[:, :, 1:].flatten()

            idx1 = np.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
            face2edge[NE0:NE1, 2] = idx1[:, :-1, :].flatten()
            face2edge[NE0:NE1, 3] = idx1[:, 1:, :].flatten()

            # y direction
            NE0 = NE1
            NE1 += nx * (ny + 1) * nz
            idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
            face2edge[NE0:NE1, 0] = idx0[:, :, :-1].flatten()
            face2edge[NE0:NE1, 1] = idx0[:, :, 1:].flatten()

            idx1 = np.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
            face2edge[NE0:NE1, 2] = idx1[:-1, :, :].flatten()
            face2edge[NE0:NE1, 3] = idx1[1:, :, :].flatten()

            # z direction
            NE0 = NE1
            NE1 += nx * ny * (nz + 1)
            idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
            face2edge[NE0:NE1, 0] = idx0[:, :-1, :].flatten()
            face2edge[NE0:NE1, 1] = idx0[:, 1:, :].flatten()

            idx1 = np.arange(nx * (ny + 1) * (nz + 1), NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
            face2edge[NE0:NE1, 2] = idx1[:-1, :, :].flatten()
            face2edge[NE0:NE1, 3] = idx1[1:, :, :].flatten()

            return face2edge
        elif bm.backend_name == 'jax':
            # x direction
            NE0 = 0
            NE1 = (nx + 1) * ny * nz
            idx0 = np.arange(nx * (ny + 1) * (nz + 1), NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
            idx1 = np.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
            
            face2edge = face2edge.at[NE0:NE1, 0].set(idx0[:, :, :-1].flatten())
            face2edge = face2edge.at[NE0:NE1, 1].set(idx0[:, :, 1:].flatten())
            face2edge = face2edge.at[NE0:NE1, 2].set(idx1[:, :-1, :].flatten())
            face2edge = face2edge.at[NE0:NE1, 3].set(idx1[:, 1:, :].flatten())

            # y direction
            NE0 = NE1
            NE1 += nx * (ny + 1) * nz
            idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
            face2edge = face2edge.at[NE0:NE1, 0].set(idx0[:, :, :-1].flatten())
            face2edge = face2edge.at[NE0:NE1, 1].set(idx0[:, :, 1:].flatten())

            idx1 = np.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
            face2edge = face2edge.at[NE0:NE1, 2].set(idx1[:-1, :, :].flatten())
            face2edge = face2edge.at[NE0:NE1, 3].set(idx1[1:, :, :].flatten())

            # z direction
            NE0 = NE1
            NE1 += nx * ny * (nz + 1)
            idx0 = np.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
            face2edge = face2edge.at[NE0:NE1, 0].set(idx0[:, :-1, :].flatten())
            face2edge = face2edge.at[NE0:NE1, 1].set(idx0[:, 1:, :].flatten())

            idx1 = np.arange(nx * (ny + 1) * (nz + 1), NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
            face2edge = face2edge.at[NE0:NE1, 2].set(idx1[:-1, :, :].flatten())
            face2edge = face2edge.at[NE0:NE1, 3].set(idx1[1:, :, :].flatten())

            return face2edge
        else:
            raise NotImplementedError("Backend is not yet implemented.")

    def face_to_cell(self) -> TensorLike:
        """
        @brief 面和单元的邻接关系, 储存每个面相邻的 2 个单元的编号
        Notes:
        - The first and second columns store the indices of the left and right cells adjacent to each face. 
        When the two indices are the same, it indicates that the face is a boundary face.
        - The third and fourth columns store the local indices of the face in the left and right cells, respectively.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        NC = self.NC

        face2cell = bm.zeros((NF, 4), dtype=self.itype)

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            # x direction
            NF0 = 0
            NF1 = (nx+1) * ny * nz
            idx = bm.arange(NC).reshape(nx, ny, nz)
            face2cell[NF0:NF1-ny*nz, 0] = idx.flatten()
            face2cell[NF0+ny*nz:NF1, 1] = idx.flatten()
            face2cell[NF0:NF1-ny*nz, 2] = 0
            face2cell[NF0:NF1-ny*nz, 3] = 1

            face2cell[NF1-ny*nz:NF1, 0] = idx[-1].flatten()
            face2cell[NF0:NF0+ny*nz, 1] = idx[0].flatten()
            face2cell[NF1-ny*nz:NF1, 2] = 1
            face2cell[NF0:NF0+ny*nz, 3] = 0

            # y direction
            idy = bm.swapaxes(idx, 1, 0)
            NF0 = NF1
            NF1 += nx * (ny+1) * nz
            fidy = bm.arange(NF0, NF1).reshape(nx, ny+1, nz).swapaxes(0, 1)
            face2cell[fidy[:-1], 0] = idy
            face2cell[fidy[1:], 1] = idy
            face2cell[fidy[:-1], 2] = 0
            face2cell[fidy[1:], 3] = 1

            face2cell[fidy[-1], 0] = idy[-1]
            face2cell[fidy[0], 1] = idy[0]
            face2cell[fidy[-1], 2] = 1
            face2cell[fidy[0], 3] = 0

            # z direction
            idz = bm.transpose(idx, (2, 0, 1))
            NF0 = NF1
            NF1 += nx * ny * (nz + 1)
            fidz = np.arange(NF0, NF1).reshape(nx, ny, nz+1).transpose(2, 0, 1)
            face2cell[fidz[:-1], 0] = idz
            face2cell[fidz[1:], 1] = idz
            face2cell[fidz[:-1], 2] = 0
            face2cell[fidz[1:], 3] = 1

            face2cell[fidz[-1], 0] = idz[-1]
            face2cell[fidz[0], 1] = idz[0]
            face2cell[fidz[-1], 2] = 1
            face2cell[fidz[0], 3] = 0

            return face2cell
        elif bm.backend_name == 'jax':
            # x direction
            NF0 = 0
            NF1 = (nx+1) * ny * nz
            idx = bm.arange(NC).reshape(nx, ny, nz)
            face2cell = face2cell.at[NF0:NF1-ny*nz, 0].set(idx.flatten())
            face2cell = face2cell.at[NF0+ny*nz:NF1, 1].set(idx.flatten())
            face2cell = face2cell.at[NF0:NF1-ny*nz, 2].set(0)
            face2cell = face2cell.at[NF0:NF1-ny*nz, 3].set(1)

            face2cell = face2cell.at[NF1-ny*nz:NF1, 0].set(idx[-1].flatten())
            face2cell = face2cell.at[NF0:NF0+ny*nz, 1].set(idx[0].flatten())
            face2cell = face2cell.at[NF1-ny*nz:NF1, 2].set(1)
            face2cell = face2cell.at[NF0:NF0+ny*nz, 3].set(0)

            # y direction
            idy = bm.swapaxes(idx, 1, 0)
            NF0 = NF1
            NF1 += nx * (ny+1) * nz
            fidy = bm.arange(NF0, NF1).reshape(nx, ny+1, nz).swapaxes(0, 1)
            face2cell = face2cell.at[fidy[:-1], 0].set(idy)
            face2cell = face2cell.at[fidy[1:], 1].set(idy)
            face2cell = face2cell.at[fidy[:-1], 2].set(0)
            face2cell = face2cell.at[fidy[1:], 3].set(1)

            face2cell = face2cell.at[fidy[-1], 0].set(idy[-1])
            face2cell = face2cell.at[fidy[0], 1].set(idy[0])
            face2cell = face2cell.at[fidy[-1], 2].set(1)
            face2cell = face2cell.at[fidy[0], 3].set(0)

            # z direction
            idz = bm.transpose(idx, (2, 0, 1))
            NF0 = NF1
            NF1 += nx * ny * (nz + 1)
            fidz = np.arange(NF0, NF1).reshape(nx, ny, nz+1).transpose(2, 0, 1)
            face2cell = face2cell.at[fidz[:-1], 0].set(idz)
            face2cell = face2cell.at[fidz[1:], 1].set(idz)
            face2cell = face2cell.at[fidz[:-1], 2].set(0)
            face2cell = face2cell.at[fidz[1:], 3].set(1)

            face2cell = face2cell.at[fidz[-1], 0].set(idz[-1])
            face2cell = face2cell.at[fidz[0], 1].set(idz[0])
            face2cell = face2cell.at[fidz[-1], 2].set(1)
            face2cell = face2cell.at[fidz[0], 3].set(0)

            return face2cell
        
        else:
            raise NotImplementedError("Backend is not yet implemented.")


    # 实体几何
    def entity_measure(self, etype: Union[int, str], index: Index = _S) -> Union[Tuple, int]:
        """
        @brief Get the measure of the entities of the specified type.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        NC = self.number_of_cells()
        if etype == 0:
            # Measure of vertices (points) is 0
            return bm.tensor(0, dtype=self.ftype)
        elif etype == 1:
            # Measure of edges, assuming edges are along x, y, z directions
            temp1 = bm.tensor([[self.h[0]], [self.h[1]], [self.h[2]]], dtype=self.ftype)
            temp2 = bm.broadcast_to(temp1, (3, int(self.NE/3)))
            return temp2.reshape(-1)
        elif etype == 2:
            # Measure of faces, assuming faces are aligned with the coordinate planes
            temp1 = bm.tensor([self.h[0] * self.h[1], self.h[0] * self.h[2], self.h[1] * self.h[2]], dtype=self.ftype)
            temp2 = bm.broadcast_to(temp1[:, None], (3, int(self.NF/3)))
            return temp2.reshape(-1)
        elif etype == 3:
            # Measure of cells (volumes)
            temp = bm.tensor(self.h[0] * self.h[1] * self.h[2], dtype=self.ftype)
            return bm.broadcast_to(temp, (NC,))
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        
    def cell_barycenter(self) -> TensorLike:
        '''
        @brief Calculate the barycenter coordinates of the cells.
        '''
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        nz = self.nz
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx-1)*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny-1)*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz-1)*self.h[2]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny)
        z = bm.linspace(box[4], box[5], nz)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij')
        bc = bm.zeros((nx, ny, nz, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc
    
    def face_barycenter(self):
        """
        @brief Calculate the coordinates range for the face centers.
        """
        xbc = self.facex_barycenter()
        ybc = self.facey_barycenter()
        zbc = self.facez_barycenter()

        return xbc, ybc, zbc

    def facex_barycenter(self):
        """
        @brief Calculate the coordinates range for the face centers in the x-direction.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        GD = self.geo_dimension()
        box = [self.origin[0],               self.origin[0] + nx*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
        x = bm.linspace(box[0], box[1], nx + 1)
        y = bm.linspace(box[2], box[3], ny)
        z = bm.linspace(box[4], box[5], nz)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij') 
        bc = bm.zeros((nx + 1, ny, nz, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc
    
    def facey_barycenter(self):
        """
        @brief Calculate the coordinates range for the face centers in the y-direction.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        GD = self.geo_dimension()
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
               self.origin[1],               self.origin[1] + ny*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny + 1)
        z = bm.linspace(box[4], box[5], nz)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij') 
        bc = bm.zeros((nx, ny + 1, nz, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc

    def facez_barycenter(self):
        """
        @brief Calculate the coordinates range for the face centers in the z-direction.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        GD = self.geo_dimension()
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny)
        z = bm.linspace(box[4], box[5], nz + 1)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij') 
        bc = bm.zeros((nx, ny, nz + 1, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc

    def edge_barycenter(self):
        """
        @brief Calculate the coordinates range for the edge centers.
        """
        xbc = self.edgex_barycenter()
        ybc = self.edgey_barycenter()
        zbc = self.edgez_barycenter()

        return xbc, ybc, zbc

    def edgex_barycenter(self):
        """
        @brief Calculate the coordinates range for the edge centers in the x-direction.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        GD = self.geo_dimension()
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        x = bm.linspace(box[0], box[1], nx)
        y = bm.linspace(box[2], box[3], ny + 1)
        z = bm.linspace(box[4], box[5], nz + 1)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij') 
        bc = bm.zeros((nx, ny + 1, nz + 1, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc
    
    def edgey_barycenter(self):
        """
        @brief Calculate the coordinates range for the edge centers in the y-direction.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        GD = self.geo_dimension()
        box = [self.origin[0], self.origin[0] + nx*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        x = bm.linspace(box[0], box[1], nx + 1)
        y = bm.linspace(box[2], box[3], ny)
        z = bm.linspace(box[4], box[5], nz + 1)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij') 
        bc = bm.zeros((nx + 1, ny, nz + 1, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc

    def edgez_barycenter(self):
        """
        @brief Calculate the coordinates range for the edge centers in the z-direction.
        """
        nx = self.nx
        ny = self.ny
        nz = self.nz
        GD = self.geo_dimension()
        box = [self.origin[0], self.origin[0] + nx*self.h[0],
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
        x = bm.linspace(box[0], box[1], nx + 1)
        y = bm.linspace(box[2], box[3], ny + 1)
        z = bm.linspace(box[4], box[5], nz)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij') 
        bc = bm.zeros((nx + 1, ny + 1, nz, GD), dtype=self.ftype)
        bc = bm.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1)

        return bc
    
    def bc_to_point(self, bcs: Union[Tuple, TensorLike], index=_S):
        """
        @brief Transform the barycentric coordinates of integration points
        to Cartesian coordinates on the actual mesh entities.

        Returns
            TensorLike: (NC, NQ, GD), (NF, NQ, GD) or (NE, NQ, GD)
        """
        node = self.entity('node')
        if isinstance(bcs, tuple) and len(bcs) == 3:
            cell = self.entity('cell', index)

            bcs0 = bcs[0].reshape(-1, 2)
            bcs1 = bcs[1].reshape(-1, 2)
            bcs2 = bcs[2].reshape(-1, 2)
            bcs = bm.einsum('im, jn, ko -> ijkmno', bcs0, bcs1, bcs2).reshape(-1, 8)

            p = bm.einsum('qj, cjk -> cqk', bcs, node[cell[:]])
        elif isinstance(bcs, tuple) and len(bcs) == 2:
            face = self.entity('face', index)

            bcs0 = bcs[0].reshape(-1, 2)
            bcs1 = bcs[1].reshape(-1, 2)
            bcs = bm.einsum('im, jn -> ijmn', bcs0, bcs1).reshape(-1, 4)

            p = bm.einsum('qj, fjk -> fqk', bcs, node[face[:]])
        else:
            edge = self.entity('edge', index=index)
            p = bm.einsum('qj, ejk -> eqk', bcs, node[edge]) 

        return p 
    
        
    # 插值点
    def interpolation_points(self, p: int, index: Index=_S):
        """
        @brief Generate all interpolation points of the mesh
        TODO Provide an efficient implementation that is distinct from unstructured meshes
        """
        c2ip = self.cell_to_ipoint(p)
        gp = self.number_of_global_ipoints(p)
        ipoint = bm.zeros([gp, 3], dtype=self.ftype)

        line = (bm.linspace(0, 1, p+1, endpoint=True, dtype=self.ftype)).reshape(-1, 1)
        line = bm.concatenate([1-line, line], axis=1)
        bcs = (line, line, line)

        cip = self.bc_to_point(bcs)
        ipoint[c2ip] = cip

        return ipoint

    def face_to_ipoint(self, p, index=None):
        """
        @brief 生成每个面上的插值点的全局编号
        TODO Provide an efficient implementation that is distinct from 
            unstructured meshes
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        edge = self.entity('edge')
        face = self.entity('face')
        face2edge = self.face_to_edge()
        edge2ipoint = self.edge_to_ipoint(p)

        mi = bm.repeat(bm.arange(p+1), p+1).reshape(-1, p+1)
        multiIndex0 = mi.flatten().reshape(-1, 1);
        multiIndex1 = mi.T.flatten().reshape(-1, 1);
        multiIndex = bm.concatenate([multiIndex0, multiIndex1], axis=1)

        dofidx = [0 for i in range(4)] 
        dofidx[0], = bm.nonzero(multiIndex[:, 1]==0)
        dofidx[1], = bm.nonzero(multiIndex[:, 1]==p)
        dofidx[2], = bm.nonzero(multiIndex[:, 0]==0)
        dofidx[3], = bm.nonzero(multiIndex[:, 0]==p)

        face2ipoint = bm.zeros([NF, (p+1)**2], dtype=self.itype)
        localEdge = bm.array([[0, 2], [1, 3], [0, 1], [2, 3]], dtype=self.itype)
        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            for i in range(4): #边上的自由度
                ge = face2edge[:, i]
                idx = bm.nonzero(face[:, localEdge[i, 0]] != edge[ge, 0])[0]

                face2ipoint[:, dofidx[i]] = edge2ipoint[ge] # TODO jax 不兼容
                face2ipoint[idx[:, None], dofidx[i]] = bm.flip(edge2ipoint[ge[idx]], axis=1) # TODO jax 不兼容

            indof = bm.all(multiIndex>0, axis=-1)&bm.all(multiIndex<p, axis=-1)
            face2ipoint[:, indof] = bm.arange(NN+NE*(p-1),
                    NN+NE*(p-1)+NF*(p-1)**2, dtype=self.itype).reshape(NF, -1) # TODO jax 不兼容
            
            return face2ipoint
        elif bm.backend_name == 'jax':
            for i in range(4):
                ge = face2edge[:, i]
                idx = bm.nonzero(face[:, localEdge[i, 0]] != edge[ge, 0])[0]

                face2ipoint = face2ipoint.at[:, dofidx[i]].set(edge2ipoint[ge])
                face2ipoint = face2ipoint.at[idx[:, None], dofidx[i]].set(bm.flip(edge2ipoint[ge[idx]], axis=1))

            indof = bm.all(multiIndex>0, axis=-1)&bm.all(multiIndex<p, axis=-1)
            face2ipoint = face2ipoint.at[:, indof].set(bm.arange(NN+NE*(p-1),
                    NN+NE*(p-1)+NF*(p-1)**2, dtype=self.itype).reshape(NF, -1))
            
            return face2ipoint
        else:
            raise NotImplementedError("Backend is not yet implemented.")
    
    def cell_to_ipoint(self, p, index=_S):
        """
        @brief 生成每个单元上的插值点的全局编号
        TODO Provide an efficient implementation that is distinct from 
            unstructured meshes
        """

        cell = self.entity('cell', index=index)
        if p == 1:
            return cell[:]

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        cell2face = self.cell_to_face()
        face2edge = self.face_to_edge()
        cell2edge = self.cell_to_edge()

        face2ipoint = self.face_to_ipoint(p)

        shape = (p+1, p+1, p+1)
        mi    = bm.arange(p+1)
        multiIndex0 = bm.broadcast_to(mi[:, None, None], shape).reshape(-1, 1)
        multiIndex1 = bm.broadcast_to(mi[None, :, None], shape).reshape(-1, 1)
        multiIndex2 = bm.broadcast_to(mi[None, None, :], shape).reshape(-1, 1)

        multiIndex = bm.concatenate([multiIndex0, multiIndex1, multiIndex2], axis=-1)

        dofidx = bm.zeros((6, (p+1)**2), dtype=self.itype) #四条边上自由度的局部编号

        dofidx[4], = bm.nonzero(multiIndex[:, 2]==0)
        dofidx[5], = bm.nonzero(multiIndex[:, 2]==p)
        dofidx[0], = bm.nonzero(multiIndex[:, 0]==0)
        dofidx[1], = bm.nonzero(multiIndex[:, 0]==p)
        dofidx[2], = bm.nonzero(multiIndex[:, 1]==0)
        dofidx[3], = bm.nonzero(multiIndex[:, 1]==p)


        cell2ipoint = bm.zeros([NC, (p+1)**3], dtype=self.itype)
        lf2e = bm.array([[4, 9, 5, 8], [6, 11, 7, 10],
                         [0, 10, 1, 8], [2, 11, 3, 9],
                         [0, 6, 2, 4], [1, 7, 3, 5]], dtype=self.itype)

        multiIndex2d = multiIndex[:(p+1)**2, 1:]
        multiIndex2d = bm.concatenate([multiIndex2d, p-multiIndex2d], axis=-1)

        lf2e = lf2e[:, [3, 0, 1, 2]]
        face2edge = face2edge[:, [2, 0, 3, 1]]

        # TODO: Provide a unified implementation that is not backend-specific
        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            for i in range(6): #面上的自由度
                gfe = face2edge[cell2face[:, i]]
                lfe = cell2edge[:, lf2e[i]]
                idx0 = bm.argsort(gfe, axis=-1)
                idx1 = bm.argsort(lfe, axis=-1)
                idx1 = bm.argsort(idx1, axis=-1)
                idx0 = idx0[bm.arange(NC)[:, None], idx1] #(NC, 4)
                idx = multiIndex2d[:, idx0].swapaxes(0, 1) #(NC, NQ, 4)

                idx = idx[..., 0]*(p+1)+idx[..., 1]
                cell2ipoint[:, dofidx[i]] = face2ipoint[cell2face[:, i, None], idx]

            indof = bm.all(multiIndex>0, axis=-1)&bm.all(multiIndex<p, axis=-1)
            cell2ipoint[:, indof] = bm.arange(NN+NE*(p-1)+NF*(p-1)**2,
                    NN+NE*(p-1)+NF*(p-1)**2+NC*(p-1)**3).reshape(NC, -1)
            
            return cell2ipoint[index]
        elif bm.backend_name == 'jax':
            for i in range(6):
                gfe = face2edge[cell2face[:, i]]
                lfe = cell2edge[:, lf2e[i]]
                idx0 = bm.argsort(gfe, axis=-1)
                idx1 = bm.argsort(lfe, axis=-1)
                idx1 = bm.argsort(idx1, axis=-1)
                idx0 = idx0[bm.arange(NC)[:, None], idx1]
                idx = multiIndex2d[:, idx0].swapaxes(0, 1)

                idx = idx[..., 0]*(p+1)+idx[..., 1]
                cell2ipoint = cell2ipoint.at[:, dofidx[i]].set(face2ipoint[cell2face[:, i, None], idx])

            indof = bm.all(multiIndex>0, axis=-1)&bm.all(multiIndex<p, axis=-1)
            cell2ipoint = cell2ipoint.at[:, indof].set(bm.arange(NN+NE*(p-1)+NF*(p-1)**2,
                    NN+NE*(p-1)+NF*(p-1)**2+NC*(p-1)**3).reshape(NC, -1))
            
            return cell2ipoint[index]
        else:
            raise NotImplementedError("Backend is not yet implemented.")
         

    # 形函数
    def jacobi_matrix(self, bcs: TensorLike, index :Index=_S) -> TensorLike:
        """
        @brief Compute the Jacobi matrix for the mapping from the reference element 
            (xi, eta, zeta) to the actual Lagrange hexahedron (x, y, z)

        x(xi, eta, zeta) = phi_0(xi, eta, zeta) * x_0 + phi_1(xi, eta, zeta) * x_1 + 
                    ... + phi_{ldof-1}(xi, eta, zeta) * x_{ldof-1}

        """
        assert isinstance(bcs, tuple)

        TD = len(bcs)
        node = self.entity('node')
        cell = self.entity('cell')
        gphi = self.grad_shape_function(bcs, p=1, variables='u')
        J = bm.einsum( 'cim, qin -> qcmn', node[cell[:]], gphi)

        return J
    

    # 其他方法
    def quadrature_formula(self, q: int, etype:Union[int, str]='cell'):
        """
        @brief Get the quadrature formula for numerical integration.
        """
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q)
        if etype == 3:
            return TensorProductQuadrature((qf, qf, qf))
        elif etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return qf
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def uniform_refine(self, n: int=1):
        """
        @brief Uniformly refine the 3D structured mesh.

        Note:
        clear method is used at the end to clear the cache of entities. This is necessary because even after refinement, 
        the entities remain the same as before refinement due to the caching mechanism.
        Structured mesh have their own entity generation methods, so the cache needs to be manually cleared.
        Unstructured mesh do not require this because they do not have entity generation methods.
        """
        for i in range(n):
            self.extent = [i * 2 for i in self.extent]
            self.h = [h / 2.0 for h in self.h]
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]
            self.nz = self.extent[5] - self.extent[4]

            self.NN = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            self.NE = (self.nx + 1) * (self.ny + 1) * self.nz + \
                    (self.nx + 1) * (self.ny + 1) * self.nz + \
                    self.nx * (self.ny + 1) * (self.nz + 1)
            self.NF = self.nx * self.ny * (self.nz + 1) + \
                    self.nx * (self.ny + 1) * self.nz + \
                    (self.nx + 1) * self.ny * self.nz
            self.NC = self.nx * self.ny * self.nz

            self.cell2edge = self.cell_to_edge()
            self.cell2face = self.cell_to_face()
            self.face2edge = self.face_to_edge()
            self.face2cell = self.face_to_cell()

        self.clear()


UniformMesh3d.set_ploter('3d')
