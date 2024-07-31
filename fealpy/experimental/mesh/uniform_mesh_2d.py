import numpy as np 
from typing import Union, Optional, Sequence, Tuple, Any

from .utils import entitymethod, estr2dim

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import StructuredMesh

class UniformMesh2d(StructuredMesh):
    def __init__(self,
                 extent: Tuple[int, int, int, int],
                 h: Tuple[float, float] = (1.0, 1.0),
                 origin: Tuple[float, float] = (0.0, 0.0)):
        super().__init__(TD=2)
        # Mesh properties
        self.extent: Tuple[int, int, int, int] = extent
        self.h: Tuple[float, float] = h
        print("h00", self.h[0])
        self.origin: Tuple[float, float] = origin

        # Mesh dimensions
        self.nx = self.extent[1] - self.extent[0]
        self.ny = self.extent[3] - self.extent[2]
        self.NN = (self.nx + 1) * (self.ny + 1)
        self.NE = self.ny * (self.nx + 1) + self.nx * (self.ny + 1)
        self.NF = self.NE
        self.NC = self.nx * self.ny

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

        self.meshtype = 'UniformMesh2d'

        self.face_to_ipoint = self.edge_to_ipoint


    @entitymethod(0)
    def _get_node(self):
        GD = 2
        nx = self.nx
        ny = self.ny
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        x = bm.linspace(box[0], box[1], nx + 1)
        y = bm.linspace(box[2], box[3], ny + 1)
        xx, yy = bm.meshgrid(x, y, indexing='ij')
        node = bm.zeros((nx + 1, ny + 1, GD), dtype=bm.float64)
        #node[..., 0] = xx
        #node[..., 1] = yy
        node = bm.concatenate((xx[..., np.newaxis], yy[..., np.newaxis]), axis=-1)

        return node
    
    @entitymethod(1)
    def _get_edge(self):
        nx = self.nx
        ny = self.ny

        NN = self.NN
        NE = self.NE

        idx = bm.arange(NN, dtype=self.itype).reshape(nx + 1, ny + 1)
        edge = bm.zeros((NE, 2), dtype=bm.int32)

        if bm.backend_name == 'numpy' or bm.backend_name == 'pytorch':
            NE0 = 0
            NE1 = nx * (ny + 1)
            edge[NE0:NE1, 0] = idx[:-1, :].reshape(-1)
            edge[NE0:NE1, 1] = idx[1:, :].reshape(-1)
            edge[NE0 + ny:NE1:ny + 1, :] = bm.flip(edge[NE0 + ny:NE1:ny + 1], axis=[1])

            #edge[NE0 + ny:NE1:ny + 1, :] = edge[NE0 + ny:NE1:ny + 1, -1::-1]

            NE0 = NE1
            NE1 += ny * (nx + 1)
            edge[NE0:NE1, 0] = idx[:, :-1].reshape(-1)
            edge[NE0:NE1, 1] = idx[:, 1:].reshape(-1)
            edge[NE0:NE0 + ny, :] = bm.flip(edge[NE0:NE0 + ny], axis=[1])

            #edge[NE0:NE0 + ny, :] = edge[NE0:NE0 + ny, -1::-1]

            return edge
        elif bm.backend_name == 'jax':
            raise NotImplementedError("Jax backend is not yet implemented.")
        else:
            raise NotImplementedError("Backend is not yet implemented.")
        
    
    @entitymethod(2)
    def _get_cell(self):
        nx = self.nx
        ny = self.ny

        NN = self.NN
        NC = self.NC
        cell = bm.zeros((NC, 4), dtype=bm.int32)
        idx = bm.arange(NN).reshape(nx + 1, ny + 1)
        c = idx[:-1, :-1]
        #cell[:, 0] = c.reshape(-1)
        #cell[:, 1] = cell[:, 0] + 1
        #cell[:, 2] = cell[:, 0] + ny + 1
        #cell[:, 3] = cell[:, 2] + 1
        cell_0 = c.reshape(-1)
        cell_1 = cell_0 + 1
        cell_2 = cell_0 + ny + 1
        cell_3 = cell_2 + 1
        cell = bm.concatenate([cell_0[:, None], cell_1[:, None], cell_2[:, None], cell_3[:, None]], axis=-1)

        return cell

    # entity
    def entity_measure(self, etype: Union[int, str]) -> TensorLike:
       if isinstance(etype, str):
           etype = estr2dim(self, etype)
       if etype == 0:
           return bm.tensor(0, dtype=bm.float64)
       elif etype == 1:
           print("h0", self.h[0])
           return self.h[0], self.h[1]
       elif etype == 2:
           return self.h[0] * self.h[1]
       else:
           raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    # def interpolation_points(self, p, index: Index=_S):
    #     cell = self.entity('cell')
    #     node = self.entity('node')
    #     if p <= 0:
    #         raise ValueError("p must be a integer larger than 0.")
    #     if p == 1:
    #         return node

    #     NN = self.number_of_nodes()
    #     GD = self.geo_dimension()

    #     gdof = self.number_of_global_ipoints(p)
    #     ipoints = np.zeros((gdof, GD), dtype=self.ftype)
    #     ipoints[:NN, :] = node

    #     NE = self.number_of_edges()

    #     edge = self.entity('edge')

    #     multiIndex = self.multi_index_matrix(p, 1)
    #     w = multiIndex[1:-1, :] / p
    #     ipoints[NN:NN + (p-1) * NE, :] = np.einsum('ij, ...jm -> ...im', w,
    #             node[edge,:]).reshape(-1, GD)

    #     w = np.einsum('im, jn -> ijmn', w, w).reshape(-1, 4)
    #     ipoints[NN + (p-1) * NE:, :] = np.einsum('ij, kj... -> ki...', w,
    #             node[cell[:]]).reshape(-1, GD)

    #     return ipoints
       
    def uniform_refine(self, n=1):
        # TODO: There is a problem with this code
        for i in range(n):
            self.extent = [i * 2 for i in self.extent]
            self.h = [h / 2.0 for h in self.h]
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]

            self.NC = self.nx * self.ny
            self.NF = self.NE
            self.NE = self.ny * (self.nx + 1) + self.nx * (self.ny + 1)
            self.NN = (self.nx + 1) * (self.ny + 1)
        
        del self.node
        del self.edge
        del self.cell
        # TODO: Implement cache clearing mechanism



