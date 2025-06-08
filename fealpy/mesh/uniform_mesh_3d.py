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
    def __init__(self, extent: Tuple[int, int, int, int, int, int] = (0, 1, 0, 1, 0, 1), 
             h: Tuple[float, float, float] = (1.0, 1.0, 1.0), 
             origin: Tuple[float, float, float] = (0.0, 0.0, 0.0), 
             ipoints_ordering='zyx', 
             flip_direction=None, 
             *, itype=None, ftype=None, device=None):
        """
        Initializes a 3D uniform structured mesh.

        Parameters:
        extent : tuple of int
            Defines the number of cells in the mesh divisions.
        h : tuple of float, optional
            Defines the step size in the x, y, and z directions.
        origin : tuple of float, optional
            Specifies the coordinates of the origin of the mesh. 
        ipoints_ordering : str, optional
            Specifies the ordering of interpolation points in the mesh. 
        flip_direction : str or None, optional
            Specifies whether to flip the direction of node numbering.
        itype : data type, optional
            Data type for integer values used in the mesh. Default is None, which is assigned as bm.int32.
        ftype : data type, optional
            Data type for floating-point values used in the mesh. Default is None, which is assigned as bm.float64.
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
        super().__init__(TD=3, itype=itype, ftype=ftype)

        self.device = device

        # Mesh properties
        # self.extent = bm.array(extent, dtype=itype, device=device)
        self.extent = extent
        self.h = bm.array(h, dtype=ftype, device=device) 
        self.origin = bm.array(origin, dtype=ftype, device=device)
        self.shape = (
                self.extent[1] - self.extent[0], 
                self.extent[3] - self.extent[2],
                self.extent[5] - self.extent[4]
                )

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

        # Interpolation points
        if ipoints_ordering not in ['zyx', 'nefc']:
            raise ValueError("The ipoints_ordering parameter must be either 'zyx' or 'nefc'")
        self.ipoints_ordering = ipoints_ordering

        # Whether to flip
        self.flip_direction = flip_direction

        # Initialize face adjustment mask
        self.adjusted_face_mask = self.get_adjusted_face_mask()

        # Specify the counterclockwise drawing
        self.ccw = bm.array([0, 2, 3, 1], dtype=self.itype)

        self.cell2edge = self.cell_to_edge()
        self.cell2face = self.cell_to_face()
        self.face2edge = self.face_to_edge()
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

    def interpolate(self, u, etype=0, keepdim=False) -> TensorLike:
        """
        Compute the interpolation of a function u on the mesh.

        Parameters:
            u: The function to be interpolated.
            etype: The type of entity on which to interpolate.

        Example:
        ```
            from fealpy.mesh import UniformMesh2d
            mesh = UniformMesh2d(
                            extent=[0, 10, 0, 10, 0, 10], 
                            h=(0.1, 0.1, 0.1),
                            origin=(0.0, 0.0, 0.0))
            u = mesh.interpolate(
                            lambda x: x[..., 0]**2 + x[..., 1]**2 + x[..., 2]**2
                            )
            print(u)
        ```
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            node = self.entity('node')
            return u(node)
        else:
            raise ValueError(f"Unsupported entity type: {etype}")

    def linear_index_map(self, etype: Union[int, str]=0):
        """
        Build and return the tensor mapping multi-dimensional 
        indices to linear indices.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return bm.arange(
                    self.NN, 
                    dtype=self.itype, 
                    device=self.device
                    ).reshape(self.nx + 1, self.ny + 1, self.nz + 1)
        elif etype == 3:
            return bm.arange(
                    self.NC, 
                    dtype=self.itype, 
                    device=self.device).reshape(self.nx, self.ny, self.nz)

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

        x = bm.linspace(box[0], box[1], nx + 1, dtype=self.ftype, device=self.device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=self.ftype, device=self.device)
        z = bm.linspace(box[4], box[5], nz + 1, dtype=self.ftype, device=self.device)
        xx, yy, zz = bm.meshgrid(x, y, z, indexing='ij')
        node = bm.concatenate((xx[..., None], yy[..., None], zz[..., None]), axis=-1)

        if self.flip_direction == 'y':
            node = bm.flip(node.reshape(nx + 1, ny + 1, nz + 1, GD), axis=1).reshape(-1, GD)
        elif self.flip_direction == 'z':
            node = bm.flip(node.reshape(nx + 1, ny + 1, nz + 1, GD), axis=2).reshape(-1, GD)

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

        idx = bm.arange(NN, dtype=self.itype, device=self.device).reshape(nx + 1, ny + 1, nz + 1)
        edge = bm.zeros((NE, 2), dtype=self.itype, device=self.device)

        NE0 = 0
        NE1 = nx * (ny + 1) * (nz + 1)
        # c = bm.transpose(idx, (0, 1, 2))[:-1, :, :]
        c = bm.permute_dims(idx, axes=(0, 1, 2))[:-1, :, :]
        edge = bm.set_at(edge, (slice(NE0, NE1), 0), c.flatten())
        edge = bm.set_at(edge, (slice(NE0, NE1), 1), edge[NE0:NE1, 0] + (ny + 1) * (nz + 1))
        
        NE0 = NE1
        NE1 += (nx + 1) * ny * (nz + 1)
        # c = bm.transpose(idx, (0, 1, 2))[:, :-1, :]
        c = bm.permute_dims(idx, axes=(0, 1, 2))[:, :-1, :]
        edge = bm.set_at(edge, (slice(NE0, NE1), 0), c.flatten())
        edge = bm.set_at(edge, (slice(NE0, NE1), 1), edge[NE0:NE1, 0] + (nz + 1))

        NE0 = NE1
        NE1 += (nx + 1) * (ny + 1) * nz
        # c = bm.transpose(idx, (0, 1, 2))[:, :, :-1]
        c = bm.permute_dims(idx, axes=(0, 1, 2))[:, :, :-1]
        edge = bm.set_at(edge, (slice(NE0, NE1), 0), c.flatten())
        edge = bm.set_at(edge, (slice(NE0, NE1), 1), edge[NE0:NE1, 0] + 1)

        return edge

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
        idx = bm.arange(NN, device=self.device).reshape(nx + 1, ny + 1, nz + 1)
        face = bm.zeros((NF, 4), dtype=self.itype, device=self.device)

        NF0 = 0
        NF1 = (nx + 1) * ny * nz
        c = idx[:, :-1, :-1]
        face = bm.set_at(face, (slice(NF0, NF1), 0), c.flatten())
        face = bm.set_at(face, (slice(NF0, NF1), 1), face[NF0:NF1, 0] + 1)
        face = bm.set_at(face, (slice(NF0, NF1), 2), face[NF0:NF1, 0] + nz + 1)
        face = bm.set_at(face, (slice(NF0, NF1), 3), face[NF0:NF1, 2] + 1)

        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        # c = bm.transpose(idx, (0, 1, 2))[:-1, :, :-1]
        c = bm.permute_dims(idx, axes=(0, 1, 2))[:-1, :, :-1]
        face = bm.set_at(face, (slice(NF0, NF1), 0), c.flatten())
        face = bm.set_at(face, (slice(NF0, NF1), 1), face[NF0:NF1, 0] + 1)
        face = bm.set_at(face, (slice(NF0, NF1), 2), face[NF0:NF1, 0] + (ny + 1) * (nz + 1))
        face = bm.set_at(face, (slice(NF0, NF1), 3), face[NF0:NF1, 2] + 1)
        NF2 = NF0 + ny * nz
        N = nz * (ny + 1)
        idx1 = bm.zeros((nx, nz), dtype=self.itype)
        idx1 = bm.arange(NF2, NF2 + nz)
        idx1 = idx1 + bm.arange(0, N * nx, N).reshape(nx, 1)
        idx1 = idx1.flatten()

        NF0 = NF1
        NF1 += nx * ny * (nz + 1)
        # c = bm.transpose(idx, (0, 1, 2))[:-1, :-1, :]
        c = bm.permute_dims(idx, axes=(0, 1, 2))[:-1, :-1, :]
        face = bm.set_at(face, (slice(NF0, NF1), 0), c.flatten())
        face = bm.set_at(face, (slice(NF0, NF1), 1), face[NF0:NF1, 0] + nz + 1)
        face = bm.set_at(face, (slice(NF0, NF1), 2), face[NF0:NF1, 0] + (ny + 1) * (nz + 1))
        face = bm.set_at(face, (slice(NF0, NF1), 3), face[NF0:NF1, 2] + nz + 1)
        N = ny * (nz + 1)
        idx2 = bm.zeros((nx, ny), dtype=self.itype)
        idx2 = bm.arange(NF0, NF0 + ny * (nz + 1), nz + 1)
        idx2 = idx2 + bm.arange(0, N * nx, N).reshape(nx, 1)
        idx2 = idx2.flatten()

        return face

    @entitymethod(3)
    def _get_cell(self) -> TensorLike:
        """
        @brief Generate the cells in a structured mesh.
        """
        NN = self.NN
        NC = self.NC
        nx, ny, nz = self.nx, self.ny, self.nz

        idx = bm.arange(NN, device=self.device).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]

        cell = bm.zeros((NC, 8), dtype=self.itype, device=self.device)
        nyz = (ny + 1) * (nz + 1)

        cell = bm.set_at(cell, (slice(None), 0), c.flatten())
        cell = bm.set_at(cell, (slice(None), 1), cell[:, 0] + 1)
        cell = bm.set_at(cell, (slice(None), 2), cell[:, 0] + nz + 1)
        cell = bm.set_at(cell, (slice(None), 3), cell[:, 2] + 1)
        cell = bm.set_at(cell, (slice(None), 4), cell[:, 0] + nyz)
        cell = bm.set_at(cell, (slice(None), 5), cell[:, 4] + 1)
        cell = bm.set_at(cell, (slice(None), 6), cell[:, 2] + nyz)
        cell = bm.set_at(cell, (slice(None), 7), cell[:, 6] + 1)
        # cell[:, 0] = c.flatten()
        # cell[:, 1] = cell[:, 0] + 1
        # cell[:, 2] = cell[:, 0] + nz + 1
        # cell[:, 3] = cell[:, 2] + 1
        # cell[:, 4] = cell[:, 0] + nyz
        # cell[:, 5] = cell[:, 4] + 1
        # cell[:, 6] = cell[:, 2] + nyz
        # cell[:, 7] = cell[:, 6] + 1

        return cell
    
    # 实体拓扑
    def number_of_nodes_of_cells(self):
        return 8

    def number_of_edges_of_cells(self):
        return 12

    def number_of_faces_of_cells(self):
        return 6
    
    def cell_to_edge(self) -> TensorLike:
        """储存每个单元相邻的 12 条边的编号"""
        NC = self.NC

        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2edge = bm.zeros((NC, 12), dtype=self.itype, device=self.device)

        # x direction
        idx0 = bm.arange(nx * (ny + 1) * (nz + 1), device=self.device).reshape(nx, ny + 1, nz + 1)
        cell2edge = bm.set_at(cell2edge, (slice(None), 0), idx0[:, :-1, :-1].flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 1), idx0[:, :-1, 1:].flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 2), idx0[:, 1:, :-1].flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 3), idx0[:, 1:, 1:].flatten())

        # y direction
        NE0 = nx * (ny + 1) * (nz + 1)
        idx1 = bm.arange((nx + 1) * ny * (nz + 1), device=self.device).reshape(nx + 1, ny, nz + 1) 
        cell2edge = bm.set_at(cell2edge, (slice(None), 4), (NE0 + idx1[:-1, :, :-1]).flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 5), (NE0 + idx1[:-1, :, 1:]).flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 6), (NE0 + idx1[1:, :, :-1]).flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 7), (NE0 + idx1[1:, :, 1:]).flatten())

        # z direction
        NE1 = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1)
        idx2 = bm.arange((nx + 1) * (ny + 1) * nz, device=self.device).reshape(nx + 1, ny + 1, nz)
        cell2edge = bm.set_at(cell2edge, (slice(None), 8), (NE1 + idx2[:-1, :-1, :]).flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 9), (NE1 + idx2[:-1, 1:, :]).flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 10), (NE1 + idx2[1:, :-1, :]).flatten())
        cell2edge = bm.set_at(cell2edge, (slice(None), 11), (NE1 + idx2[1:, 1:, :]).flatten())

        return cell2edge
        
    def cell_to_face(self, index: Index=_S) -> TensorLike:
        """
        @brief 单元和面的邻接关系, 储存每个单元相邻的六个面的编号
        """
        NC = self.NC
        nx = self.nx
        ny = self.ny
        nz = self.nz

        cell2face = bm.zeros((NC, 6), dtype=self.itype, device=self.device)

        # x direction
        idx0 = bm.arange((nx + 1) * ny * nz, device=self.device).reshape(nx + 1, ny, nz)
        cell2face = bm.set_at(cell2face, (slice(None), 0), idx0[:-1, :, :].flatten())
        cell2face = bm.set_at(cell2face, (slice(None), 1), idx0[1:, :, :].flatten())    

        # y direction
        NE0 = (nx + 1) * ny * nz
        idx1 = bm.arange(nx * (ny + 1) * nz, device=self.device).reshape(nx, ny + 1, nz)
        cell2face = bm.set_at(cell2face, (slice(None), 2), (NE0 + idx1[:, :-1, :]).flatten())
        cell2face = bm.set_at(cell2face, (slice(None), 3), (NE0 + idx1[:, 1:, :]).flatten())

        # z direction
        NE1 = (nx + 1) * ny * nz + nx * (ny + 1) * nz
        idx2 = bm.arange(nx * ny * (nz + 1), device=self.device).reshape(nx, ny, nz + 1)
        cell2face = bm.set_at(cell2face, (slice(None), 4), (NE1 + idx2[:, :, :-1]).flatten())
        cell2face = bm.set_at(cell2face, (slice(None), 5), (NE1 + idx2[:, :, 1:]).flatten())
        
        return cell2face[index]
        
    def face_to_edge(self, index: Index=_S):
        """
        @brief 面和边的邻接关系, 储存每个面相邻的 4 条边的编号
        """

        NE = self.NE
        NF = self.NF
        nx = self.nx
        ny = self.ny
        nz = self.nz
        face2edge = bm.zeros((NF, 4), dtype=self.itype)

        # x direction
        NE0 = 0
        NE1 = (nx + 1) * ny * nz
        idx0 = bm.arange(nx * (ny + 1) * (nz + 1), 
                         NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 0), idx0[:, :, :-1].flatten())
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 1), idx0[:, :, 1:].flatten())
        # face2edge[NE0:NE1, 0] = idx0[:, :, :-1].flatten()
        # face2edge[NE0:NE1, 1] = idx0[:, :, 1:].flatten()

        idx1 = bm.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 2), idx1[:, :-1, :].flatten())
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 3), idx1[:, 1:, :].flatten())
        # face2edge[NE0:NE1, 2] = idx1[:, :-1, :].flatten()
        # face2edge[NE0:NE1, 3] = idx1[:, 1:, :].flatten()

        # y direction
        NE0 = NE1
        NE1 += nx * (ny + 1) * nz
        idx0 = bm.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 0), idx0[:, :, :-1].flatten())
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 1), idx0[:, :, 1:].flatten())
        # face2edge[NE0:NE1, 0] = idx0[:, :, :-1].flatten()
        # face2edge[NE0:NE1, 1] = idx0[:, :, 1:].flatten()

        idx1 = bm.arange(NE - (nx + 1) * (ny + 1) * nz, NE).reshape(nx + 1, ny + 1, nz)
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 2), idx1[:-1, :, :].flatten())
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 3), idx1[1:, :, :].flatten())
        # face2edge[NE0:NE1, 2] = idx1[:-1, :, :].flatten()
        # face2edge[NE0:NE1, 3] = idx1[1:, :, :].flatten()

        # z direction
        NE0 = NE1
        NE1 += nx * ny * (nz + 1)
        idx0 = bm.arange(nx * (ny + 1) * (nz + 1)).reshape(nx, ny + 1, nz + 1)
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 0), idx0[:, :-1, :].flatten())
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 1), idx0[:, 1:, :].flatten())
        # face2edge[NE0:NE1, 0] = idx0[:, :-1, :].flatten()
        # face2edge[NE0:NE1, 1] = idx0[:, 1:, :].flatten()

        idx1 = bm.arange(nx * (ny + 1) * (nz + 1), 
                         NE - (nx + 1) * (ny + 1) * nz).reshape(nx + 1, ny, nz + 1)
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 2), idx1[:-1, :, :].flatten())
        face2edge = bm.set_at(face2edge, (slice(NE0, NE1), 3), idx1[1:, :, :].flatten())
        # face2edge[NE0:NE1, 2] = idx1[:-1, :, :].flatten()
        # face2edge[NE0:NE1, 3] = idx1[1:, :, :].flatten()

        return face2edge[index]

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

        # x direction
        NF0 = 0
        NF1 = (nx+1) * ny * nz
        idx = bm.arange(NC, dtype=face2cell.dtype).reshape(nx, ny, nz)
        face2cell = bm.set_at(face2cell, (slice(NF0, NF1 - ny * nz), 0), idx.flatten())
        face2cell = bm.set_at(face2cell, (slice(NF0 + ny * nz, NF1), 1), idx.flatten())
        face2cell = bm.set_at(face2cell, (slice(NF0, NF1 - ny * nz), 2), 0)
        face2cell = bm.set_at(face2cell, (slice(NF0, NF1 - ny * nz), 3), 1)

        face2cell = bm.set_at(face2cell, (slice(NF1 - ny * nz, NF1), 0), idx[-1].flatten())
        face2cell = bm.set_at(face2cell, (slice(NF0, NF0 + ny * nz), 1), idx[0].flatten())
        face2cell = bm.set_at(face2cell, (slice(NF1 - ny * nz, NF1), 2), 1)
        face2cell = bm.set_at(face2cell, (slice(NF0, NF0 + ny * nz), 3), 0)

        # y direction
        idy = bm.astype(bm.swapaxes(idx, 1, 0), face2cell.dtype)
        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        fidy = bm.arange(NF0, NF1, dtype=face2cell.dtype).reshape(nx, ny+1, nz).swapaxes(0, 1)
        face2cell = bm.set_at(face2cell, (fidy[:-1], 0), idy)
        face2cell = bm.set_at(face2cell, (fidy[1:], 1), idy)
        face2cell = bm.set_at(face2cell, (fidy[:-1], 2), 0)
        face2cell = bm.set_at(face2cell, (fidy[1:], 3), 1)

        face2cell = bm.set_at(face2cell, (fidy[-1], 0), idy[-1])
        face2cell = bm.set_at(face2cell, (fidy[0], 1), idy[0])
        face2cell = bm.set_at(face2cell, (fidy[-1], 2), 1)
        face2cell = bm.set_at(face2cell, (fidy[0], 3), 0)

        # z direction
        # idz = bm.astype(bm.transpose(idx, (2, 0, 1)), face2cell.dtype)
        idz = bm.astype(bm.permute_dims(idx, axes=(2, 0, 1)), face2cell.dtype)
        NF0 = NF1
        NF1 += nx * ny * (nz + 1)
        # NOTE 2021/09/07: The following line is incorrect. The correct line is the next one. 
        # transpose 只接受两个参数
        # fidz = bm.arange(NF0, NF1, dtype=face2cell.dtype).reshape(nx, ny, nz+1).transpose(2, 0, 1)
        fidz = bm.permute_dims(bm.arange(NF0, NF1, dtype=face2cell.dtype).reshape(nx, ny, nz+1), axes=(2, 0, 1))
        face2cell = bm.set_at(face2cell, (fidz[:-1], 0), idz)
        face2cell = bm.set_at(face2cell, (fidz[1:], 1), idz)
        face2cell = bm.set_at(face2cell, (fidz[:-1], 2), 0)
        face2cell = bm.set_at(face2cell, (fidz[1:], 3), 1)

        face2cell = bm.set_at(face2cell, (fidz[-1], 0), idz[-1])
        face2cell = bm.set_at(face2cell, (fidz[0], 1), idz[0])
        face2cell = bm.set_at(face2cell, (fidz[-1], 2), 1)
        face2cell = bm.set_at(face2cell, (fidz[0], 3), 0)

        return face2cell
        
    def boundary_node_flag(self):
        """
        @brief Determine if a point is a boundary point.
        """
        NN = self.NN
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = bm.zeros((NN,), dtype=bool)
        # isBdPoint[face[isBdFace, :]] = True
        isBdPoint = bm.set_at(isBdPoint, face[isBdFace, :], True)
        
        return isBdPoint
        
    def boundary_edge_flag(self):
        """
        @brief Determine if an edge is a boundary edge.
        """
        NE = self.NE
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = bm.zeros((NE,), dtype=bool)
        # isBdEdge[face2edge[isBdFace, :]] = True
        isBdEdge = bm.set_at(isBdEdge, face2edge[isBdFace, :], True)
        
        return isBdEdge
        
    def boundary_face_flag(self):
        """
        @brief Determine if a face is a boundary face.
        """
        face2cell = self.face_to_cell()

        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self):
        """
        @brief Determine if a cell is a boundary cell.
        """
        NC = self.NC

        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = bm.zeros((NC,), dtype=bool)
        isBdCell = bm.set_at(isBdCell, face2cell[isBdFace, 0], True)
        # isBdCell[face2cell[isBdFace, 0]] = True

        return isBdCell
        

#################################### 实体几何 #############################################
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
            temp = bm.tensor(self.h[0] * self.h[1] * self.h[2], dtype=self.ftype, device=self.device)
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

    def get_adjusted_face_mask(self) -> TensorLike:
        """
        Determine which faces need to have their direction adjusted to ensure normals point outward.

        Returns:
            TensorLike[NF]: A boolean array where True indicates that the face's direction should be adjusted.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        NF = self.NF
        adjusted_face = bm.zeros((NF,), dtype=bool)

        # Adjust faces in xy-plane
        NF0 = 0
        NF1 = (nx + 1) * ny * nz
        flip_indices_xy = bm.arange(NF0, NF0 + ny * nz)
        adjusted_face = bm.set_at(adjusted_face, flip_indices_xy, True)

        # Adjust faces in yz-plane
        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        NF2 = NF0 + ny * nz
        N = nz * (ny + 1)
        idx1 = bm.arange(NF2, NF2 + nz)
        idx1 = idx1 + bm.arange(0, N * nx, N).reshape(nx, 1)
        idx1 = idx1.flatten()
        adjusted_face = bm.set_at(adjusted_face, idx1, True)

        # Adjust faces in xz-plane
        NF0 = NF1
        NF1 += nx * ny * (nz + 1)
        N = ny * (nz + 1)
        idx2 = bm.arange(NF0, NF0 + ny * (nz + 1), nz + 1)
        idx2 = idx2 + bm.arange(0, N * nx, N).reshape(nx, 1)
        idx2 = idx2.flatten()
        adjusted_face = bm.set_at(adjusted_face, idx2, True)

        return adjusted_face

    def face_normal(self, index: Index=_S, unit: bool=False, out=None) -> TensorLike:
        """
        Calculate the normal vectors of the faces.

        Parameters:
            index (Index, optional): Index to select specific faces. 
                                        Defaults to _S (all faces).
            unit (bool, optional): If True, returns unit normal vectors. 
                                        Defaults to False.
            out (TensorLike, optional): Optional output array to store the result. 
                                        Defaults to None.

        Returns:
            TensorLike[NF, GD]: Normal vectors of the faces.
        
        法向量方向的定义如下：
            1 ---←--- 3
            |         |
            ↑         |
            |         |
            0 ------- 2
        """
        face = self.entity('face', index=index)
        node = self.entity('node')
        
        # Calculate vectors v1 and v2 for the normal calculation
        v1 = node[face[:, 1]] - node[face[:, 0]]
        v2 = node[face[:, 1]] - node[face[:, 3]]
        normals = bm.cross(v1, v2)

        adjusted_face_mask = self.get_adjusted_face_mask() 

        # Use the adjusted face mask to flip the normals if necessary
        normals = bm.set_at(normals, (adjusted_face_mask, slice(None)), 
                                -normals[adjusted_face_mask])

        # Normalize the normals if unit is True
        if unit:
            norm = bm.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / norm

        if out is not None:
            out[...] = normals
            return out

        return normals 
    
    def face_unit_normal(self, index: Index=_S, out=None) -> TensorLike:
        """
        Calculate the unit normal vectors of the faces.

        Parameters:
            index (Index, optional): Index to select specific faces. Defaults to _S (all faces).
            out (TensorLike, optional): Optional output array to store the result. Defaults to None.

        Returns:
            TensorLike[NF, GD]: Unit normal vectors of the faces.
        """
        return self.face_normal(index=index, unit=True, out=out)


#################################### 插值点 #############################################
    def interpolation_points(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Generate all interpolation points of the 3D mesh

        Ordering of interpolation points follows the sequence:
        - Z direction first, then Y direction, and finally X direction.
        '''
        if p <= 0:
            raise ValueError("p must be an integer larger than 0.")
        if p == 1:
            return self.entity('node', index=index)
        
        ordering = self.ipoints_ordering
        
        if ordering == 'zyx':
            nx = self.nx
            ny = self.ny
            nz = self.nz
            hx = self.h[0]
            hy = self.h[1]
            hz = self.h[2]

            nix = nx + 1 + nx * (p - 1)
            niy = ny + 1 + ny * (p - 1)
            niz = nz + 1 + nz * (p - 1)
            
            length_x = nx * hx
            length_y = ny * hy
            length_z = nz * hz

            ix = bm.linspace(0, length_x, nix)
            iy = bm.linspace(0, length_y, niy)
            iz = bm.linspace(0, length_z, niz)

            x, y, z = bm.meshgrid(ix, iy, iz, indexing='ij')
            ipoints = bm.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        elif ordering == 'nefc':
            c2ip = self.cell_to_ipoint(p)
            gp = self.number_of_global_ipoints(p)
            ipoints = bm.zeros([gp, 3], dtype=self.ftype)

            line = (bm.linspace(0, 1, p+1, endpoint=True, dtype=self.ftype)).reshape(-1, 1)
            line = bm.concatenate([1-line, line], axis=1)
            bcs = (line, line, line)

            cip = self.bc_to_point(bcs)
            ipoints[c2ip] = cip
        else:
            raise ValueError("Invalid ordering type. \
                    Choose 'yxz' for y-direction first, then x-direction, and finally z-direction ordering, "\
                    "or 'nec' for node first, then edge, thean face, and finally cell ordering.")
        
        return ipoints[index]
    
    
    def node_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Returns the interpolation point indices corresponding to each node in a 3D mesh.

        @param p: Interpolation order. Must be an integer greater than 0.
        @param index: Index to select specific node interpolation points.

        @return: A 1D array of size (NN,) containing the indices of interpolation points at each node.
        '''
        ordering = self.ipoints_ordering
        
        if ordering == 'zyx':
            nx = self.nx
            ny = self.ny
            nz = self.nz
            nix = nx + 1 + nx * (p - 1)
            niy = ny + 1 + ny * (p - 1)
            niz = nz + 1 + nz * (p - 1)
            
            node_x_indices = bm.arange(0, nix, p, device=self.device)
            node_y_indices = bm.arange(0, niy, p, device=self.device)
            node_z_indices = bm.arange(0, niz, p, device=self.device)
            
            node_z_grid, node_y_grid, node_x_grid = bm.meshgrid(node_z_indices, node_y_indices, node_x_indices, indexing='ij')
            
            node2ipoint = (node_z_grid * (nix * niy) + node_y_grid * nix + node_x_grid).flatten()
        
        elif ordering == 'nefc':
            NN = self.NN
            node2ipoint = bm.arange(0, NN)
        
        else:
            raise ValueError("Invalid ordering type. Choose 'zyx' or 'nefc'.")
        
        return node2ipoint[index]
    
    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Returns the interpolation point indices corresponding to each edge in a 3D mesh.

        @param p: Interpolation order. Must be an integer greater than 0.

        @return: A 2D array of size (NE, p+1) containing the indices of interpolation points at each edge.
        '''
        if p <= 0:
            raise ValueError("p must be an integer larger than 0.")
        
        ordering = self.ipoints_ordering
        edges = self.edge[index]
        
        if ordering == 'zyx':
            node_to_ipoint = self.node_to_ipoint(p)
            
            start_indices = node_to_ipoint[edges[:, 0]]
            end_indices = node_to_ipoint[edges[:, 1]]
            
            linspace_indices = bm.linspace(0, 1, p + 1, endpoint=True, dtype=self.ftype, device=self.device).reshape(1, -1)
            edge2ipoint = start_indices[:, None] * (1 - linspace_indices) + \
                          end_indices[:, None] * linspace_indices
            edge2ipoint = bm.astype(edge2ipoint, self.itype)
        elif ordering == 'nefc':
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            
            indices = bm.arange(NE, dtype=self.itype)[index]
            edge2ipoint =  bm.concatenate([
                edges[:, 0].reshape(-1, 1),
                (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, dtype=self.itype) + NN,
                edges[:, 1].reshape(-1, 1),
            ], axis=-1)
        else:
            raise ValueError("Invalid ordering type. Choose 'zyx' or 'nefc'.")
        return edge2ipoint
    
    def face_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        '''
        @brief Returns the interpolation point indices corresponding to each face in a 3D mesh.

        @param p: Interpolation order. Must be an integer greater than 0.
        @param index: Index to select specific face interpolation points.

        @return: A 2D array of size (NF, (p+1)**2) containing the indices of interpolation points at each face.
        '''
        if p <= 0:
            raise ValueError("p must be an integer larger than 0.")

        ordering = self.ipoints_ordering

        if ordering == 'zyx':
            edge_to_ipoint = self.edge_to_ipoint(p)
            face2edge = self.face_to_edge(index=index)

            start_indices = edge_to_ipoint[face2edge[:, 0]]
            end_indices = edge_to_ipoint[face2edge[:, 1]]  

            linspace_indices = bm.linspace(0, 1, p + 1, endpoint=True, dtype=self.ftype, device=self.device).reshape(1, -1)
            face_ipoints_interpolated = start_indices[:, :, None] * (1 - linspace_indices) + \
                                        end_indices[:, :, None] * linspace_indices

            # face2ipoint = face_ipoints_interpolated.reshape(-1, (p+1)**2).astype(self.itype)
            face2ipoint = bm.astype(face_ipoints_interpolated.reshape(-1, (p+1)**2), self.itype)
        elif ordering == 'nefc':
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

            for i in range(4):
                ge = face2edge[:, i]
                idx = bm.nonzero(face[:, localEdge[i, 0]] != edge[ge, 0])[0]

                face2ipoint[:, dofidx[i]] = edge2ipoint[ge]
                face2ipoint[idx[:, None], dofidx[i]] = bm.flip(edge2ipoint[ge[idx]], axis=1)

            indof = bm.all(multiIndex>0, axis=-1)&bm.all(multiIndex<p, axis=-1)
            face2ipoint[:, indof] = bm.arange(NN+NE*(p-1),
                    NN+NE*(p-1)+NF*(p-1)**2, dtype=self.itype).reshape(NF, -1)
            face2ipoint = face2ipoint[index]
            
        return face2ipoint
    
    def cell_to_ipoint(self, p, index: Index=_S):
        """
        @brief Returns the interpolation point indices corresponding to each cell in a 3D mesh.

        @param p: Interpolation order. Must be an integer greater than 0.
        @param index: Index to select specific cell interpolation points.

        @return: A 2D array of size (NC, (p+1)**3) containing the indices of interpolation points at each cell.
        """
        if p == 1:
            return self.entity('cell', index=index)
        
        ordering = self.ipoints_ordering

        if ordering == 'zyx':
            face_to_ipoint = self.face_to_ipoint(p)
            cell2face = self.cell_to_face(index=index)

            start_indices = face_to_ipoint[cell2face[:, 0]]
            end_indices = face_to_ipoint[cell2face[:, 1]]

            linspace_indices = bm.linspace(0, 1, p + 1, endpoint=True, dtype=self.ftype).reshape(1, -1)
            cell_ipoints_interpolated = start_indices[:, :, None] * (1 - linspace_indices) + \
                                        end_indices[:, :, None] * linspace_indices
            
            # 首先，转换形状以便重新排列为列优先
            reshaped = cell_ipoints_interpolated.reshape(-1, (p+1)**2, p+1)
            # 然后，转置最后两个维度，这样插值点会按照 z, y, x 的顺序排列
            transposed = reshaped.transpose(0, 2, 1)
            # 最后，reshape 到最终的形状，确保插值点是列优先排序
            cell2ipoint = transposed.reshape(-1, (p+1)**3).astype(self.itype)
        elif ordering == 'nefc':
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

            for i in range(6):
                    gfe = face2edge[cell2face[:, i]]
                    lfe = cell2edge[:, lf2e[i]]
                    idx0 = bm.argsort(gfe, axis=-1)
                    idx1 = bm.argsort(lfe, axis=-1)
                    idx1 = bm.argsort(idx1, axis=-1)
                    idx0 = idx0[bm.arange(NC)[:, None], idx1]
                    idx = multiIndex2d[:, idx0].swapaxes(0, 1)

                    idx = idx[..., 0]*(p+1)+idx[..., 1]
                    cell2ipoint[:, dofidx[i]] = face2ipoint[cell2face[:, i, None], idx]

            indof = bm.all(multiIndex>0, axis=-1)&bm.all(multiIndex<p, axis=-1)
            cell2ipoint[:, indof] = bm.arange(NN+NE*(p-1)+NF*(p-1)**2,
                    NN+NE*(p-1)+NF*(p-1)**2+NC*(p-1)**3).reshape(NC, -1)
            cell2ipoint = cell2ipoint[index]
            
        return cell2ipoint
         
    # 形函数
    def jacobi_matrix(self, bcs: TensorLike, index :Index=_S) -> TensorLike:
        """
        @brief Compute the Jacobi matrix for the mapping from the reference element 
            (xi, eta, zeta) to the actual Lagrange hexahedron (x, y, z)

        x(xi, eta, zeta) = phi_0(xi, eta, zeta) * x_0 + phi_1(xi, eta, zeta) * x_1 + 
                    ... + phi_{ldof-1}(xi, eta, zeta) * x_{ldof-1}

        """
        assert isinstance(bcs, tuple)

        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bcs, p=1, variables='u')   # (NQ, ldof, GD)
        #TODO 这里不能翻转网格，否则会导致 Jacobian 计算错误
        node_cell_flip = node[cell[:]]                             # (NC, NCN, GD)
        J = bm.einsum( 'cim, qin -> cqmn', node_cell_flip, gphi)

        return J
    
        
    # 第一基本形式
    def first_fundamental_form(self, J: TensorLike) -> TensorLike:
        """
        @brief Compute the first fundamental form from the Jacobi matrix.
        """
        TD = J.shape[-1]

        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype, device=self.device)

        for i in range(TD):
            # 计算对角元素
            diag_val = bm.einsum('...d, ...d -> ...', J[..., i], J[..., i])
            G = bm.set_at(G, (..., i, i), diag_val)
            
            for j in range(i+1, TD):
                # 计算非对角元素
                off_diag_val = bm.einsum('...d, ...d -> ...', J[..., i], J[..., j])
                G = bm.set_at(G, (..., i, j), off_diag_val)
                G = bm.set_at(G, (..., j, i), off_diag_val)

        return G  


    # 其他方法
    def quadrature_formula(self, q: int, etype:Union[int, str]='cell'):
        """
        @brief Get the quadrature formula for numerical integration.
        """
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
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
        @brief Uniformly refine the 2D structured mesh.

        Note:
        The clear method is used at the end to clear the cache of entities. 
        This is necessary because the entities remain the same as before refinement due to caching.
        Structured meshes have their own entity generation methods, so the cache needs to be manually cleared.
        Unstructured meshes do not require this because they do not have entity generation methods.
        """
        for i in range(n):
            # self.extent = 2*self.extent
            self.extent = [i * 2 for i in self.extent]
            self.h = self.h/2.0 
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

    def to_vtk(self, filename, celldata=None, nodedata=None):
        """
        @brief: Converts the 3D mesh data to a VTK structured grid format and writes to a VTS file
        """
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        if celldata is None:
            celldata = self.celldata

        if nodedata is None:
            nodedata = self.nodedata

        # 网格参数
        nx, ny, nz = self.nx, self.ny, self.nz  
        h = self.h  
        origin = self.origin

        # 创建三维坐标点
        x = bm.linspace(origin[0], origin[0] + nx * h[0], nx + 1)
        y = bm.linspace(origin[1], origin[1] + ny * h[1], ny + 1)
        z = bm.linspace(origin[2], origin[2] + nz * h[2], nz + 1) 

        # 创建三维网格点坐标
        xyz_x, xyz_y, xyz_z = bm.meshgrid(x, y, z, indexing='ij')

        if self.flip_direction == 'y':
            # 左上到右下：在 y 方向翻转
            xyz_x = xyz_x[:, ::-1, :]
            xyz_y = xyz_y[:, ::-1, :]
            xyz_z = xyz_z[:, ::-1, :]
            
            # 将坐标展平为一维数组
            points_x = xyz_x.flatten()
            points_y = xyz_y.flatten()
            points_z = xyz_z.flatten()
        elif self.flip_direction == None:
            # 默认：左下到右上
            points_x = xyz_x.flatten()
            points_y = xyz_y.flatten()
            points_z = xyz_z.flatten()

        # 创建 VTK 网格对象
        rectGrid = vtk.vtkStructuredGrid()
        # 注意：VTK 中维度顺序为 (nz+1, ny+1, nx+1)
        rectGrid.SetDimensions(nz + 1, ny + 1, nx + 1)

        # 创建点
        points = vtk.vtkPoints()
        for i in range(len(points_x)):
            points.InsertNextPoint(points_x[i], points_y[i], points_z[i])
        rectGrid.SetPoints(points)

        # 添加节点数据
        if nodedata is not None:
            for name, data in nodedata.items():
                # 如果数据是元组，取第一个元素（当前密度值）
                if isinstance(data, tuple):
                    current_data = data[0]  # 获取当前密度值
                else:
                    current_data = data

                # 确保数据在 CPU 上
                current_data = current_data.cpu() if hasattr(current_data, 'cpu') else current_data
            
                
                # 检查是否为向量场数据
                if len(current_data.shape) > 1 and current_data.shape[-1] > 1:
                    # 向量场数据
                    n_components = current_data.shape[-1]  # 获取向量维度
                    if len(current_data.shape) == 4:  # 3D网格的向量场
                        # 重塑数据为2D数组：(n_points, n_components)
                        reshaped_data = current_data.reshape(-1, n_components)
                    else:
                        reshaped_data = current_data
                    
                    data_array = numpy_to_vtk(reshaped_data, deep=True)
                    data_array.SetName(name)
                    data_array.SetNumberOfComponents(n_components)
                else:
                    # 标量场数据
                    data_array = numpy_to_vtk(current_data.flatten(), deep=True)
                    data_array.SetName(name)
                rectGrid.GetPointData().AddArray(data_array)

        # 添加单元格数据
        if celldata is not None:
            for name, data in celldata.items():
                # 如果数据是元组，取第一个元素（当前密度值）
                if isinstance(data, tuple):
                    current_data = data[0]  # 获取当前密度值
                else:
                    current_data = data

                # 确保数据在 CPU 上
                current_data = current_data.cpu() if hasattr(current_data, 'cpu') else current_data

                # 将数据展平
                data_array = numpy_to_vtk(current_data.flatten(), deep=True)
                data_array.SetName(name)
                rectGrid.GetCellData().AddArray(data_array)

        # 写入 VTK 文件
        print("Writing to vtk...")
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(rectGrid)
        writer.SetFileName(filename)
        writer.Write()

        return filename
    
    def function(self, etype='node', dtype=None, ex=0):
        """返回定义在节点、网格边、或者网格单元上离散函数 (数组), 元素取值为 0"""
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 0}:
            uh = bm.zeros((nx+1+2*ex, ny+1+2*ex, nz+1+2*ex), dtype=dtype)
        elif etype in {'facex'}: # 法线和 x 轴平行的面
            uh = bm.zeros((nx+1, ny, nz), dtype=dtype)
        elif etype in {'facey'}: # 法线和 y 轴平行的面
            uh = bm.zeros((nx, ny+1, nz), dtype=dtype)
        elif etype in {'facez'}: # 法线和 z 轴平行的面
            uh = bm.zeros((nx, ny, nz+1), dtype=dtype)
        elif etype in {'face', 2}: # 所有的面
            ex = bm.zeros((nx+1, ny, nz), dtype=dtype)
            ey = bm.zeros((nx, ny+1, nz), dtype=dtype)
            ez = bm.zeros((nx, ny, nz+1), dtype=dtype)
            uh = (ex, ey, ez)
        elif etype in {'edgex'}: # 切向与 x 轴平行的边
            uh = bm.zeros((nx, ny+1, nz+1), dtype=dtype)
        elif etype in {'edgey'}: # 切向与 y 轴平行的边
            uh = bm.zeros((nx+1, ny, nz+1), dtype=dtype)
        elif etype in {'edgez'}: # 切向与 z 轴平行的边
            uh = bm.zeros((nx+1, ny+1, nz), dtype=dtype)
        elif etype in {'edge', 1}: # 所有的边
            ex = bm.zeros((nx, ny+1, nz+1), dtype=dtype)
            ey = bm.zeros((nx+1, ny, nz+1), dtype=dtype)
            ez = bm.zeros((nx+1, ny+1, nz), dtype=dtype)
            uh = (ex, ey, ez)
        elif etype in {'cell', 3}:
            uh = bm.zeros((nx+2*ex, ny+2*ex, nz+2*ex), dtype=dtype)
        else:
            raise ValueError(f'the entity `{etype}` is not correct!')

        return uh
    
    def error(self, u, uh, errortype='all'):
        """Compute error metrics between exact and numerical solutions in 3D space.
    
        Calculates various error norms between the exact solution u(x,y,z) and the 
        numerical solution uh on a 3D grid. Supports multiple error metrics including
        maximum absolute error, continuous L2 norm (integral-based), and discrete l2 norm (average-based).

        Parameters
            u : Callable
                The exact solution function u(x,y,z) that takes node coordinates (N×3 array)
                and returns exact solution values (N×1 array). Must match the grid used for uh.
            uh : TensorLike
                The numerical solution values at discrete nodes with shape (nx+1, ny+1, nz+1).
                Should correspond to the same grid points as u(node).
            errortype : str, optional, default='all'
                Specifies which error norm(s) to compute:
                - 'all': returns all three error metrics (emax, e0, el2)
                - 'max': only maximum absolute error (L∞ norm)
                - 'L2': only continuous L2 norm error
                - 'l2': only discrete l2 norm error

        Returns
            error_metrics : Union[float, Tuple[float, float, float]]
                The computed error metric(s):
                - If 'all': returns tuple (emax, e0, el2)
                    emax: maximum absolute error (L∞ norm)
                    e0: continuous L2 norm error (hx*hy*hz weighted)
                    el2: discrete l2 norm error (average-based)
                - Otherwise returns single float for specified error type

        Raises
            AssertionError
                If uh dimensions don't match grid dimensions (nx+1, ny+1, nz+1)

        Notes
            Error norms are computed as:
            - L∞ norm: max|u(x_i,y_j,z_k) - uh(x_i,y_j,z_k)|
            - Continuous L2: sqrt(hx*hy*hz * Σ(u-uh)²)
            - Discrete l2: sqrt(1/((nx-1)(ny-1)(nz-1)) * Σ(u-uh)²)

            where:
            - hx, hy, hz are mesh spacings in x,y,z directions
            - nx, ny, nz are numbers of grid intervals in each direction
            - The grid has (nx+1)×(ny+1)×(nz+1) nodes

        Examples
            >>> # For 3D problem with 11×11×11 grid (10 intervals each direction)
            >>> exact_sol = lambda p: p[:,0]**2 + p[:,1]**2 + p[:,2]**2
            >>> numerical_sol = bm.ones((11,11,11))  # dummy solution
            >>> errors = error(exact_sol, numerical_sol)
            >>> emax, eL2, el2 = errors
        """
        # assert (uh.shape[0] == self.nx+1) and (uh.shape[1] == self.ny+1) and (uh.shape[2] == self.nz+1)
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        nx = self.nx
        ny = self.ny
        nz = self.nz
        node = self.node
        uI = u(node)
        e = uI - uh

        if errortype == 'all':
            emax = bm.max(bm.abs(e))
            e0 = bm.sqrt(hx * hy * hz * bm.sum(e ** 2))
            el2 = bm.sqrt(1 / ((nx - 1) * (ny - 1)) * (nz - 1) * bm.sum(e ** 2))
            return emax, e0, el2
        elif errortype == 'max':
            emax = bm.max(bm.abs(e))
            return emax
        elif errortype == 'L2':
            e0 = bm.sqrt(hx * hy * hz * bm.sum(e ** 2))
            return e0
        elif errortype == 'l2':
            el2 = bm.sqrt(1 / ((nx - 1) * (ny - 1)) * (nz - 1) * bm.sum(e ** 2))
            return el2

    def show_function(self, plot, uh, cmap='jet'):
        pass

    ## @ingroup GeneralInterface
    def show_animation(self, fig, axes, box,
                       init, forward, fname='test.mp4',
                       fargs=None, frames=1000, lw=2, interval=50):
        pass


UniformMesh3d.set_ploter('3d')
