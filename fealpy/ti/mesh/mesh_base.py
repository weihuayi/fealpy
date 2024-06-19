from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti

from .. import logger
from ..sparse import CSRMatrix

from .utils import EntityName, Entity, Field, Index, _S, _int_func
from .utils import estr2dim
from . import functional as F

@ti.data_oriented
class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    cell: Entity 
    face: Entity
    edge: Entity 
    node: Entity 
    face2cell: Field 
    cell2edge: Field 
    localEdge: Field # only for homogeneous mesh
    localFace: Field # only for homogeneous mesh

    def __init__(self, NN: int, TD: int) -> None: 
        self._entity_storage: Dict[int, _T] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Entity: ...
    def __getattr__(self, name: str):
        """
        """
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        return self.entity(name) 

    def __setattr__(self, name: str, value: Entity) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('Please call super().__init__() before setting attributes!')
            edim = estr2dim(self, name)
            self._entity_storage[edim] = value
        else:
            super().__setattr__(name, value) 

    ### properties
    def top_dimension(self) -> int: return self.TD

    @property
    def itype(self): return self.cell.dtype

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        edim = estr2dim(self, etype) if isinstance(etype, str) else etype
        entity = self.entity(edim) 
        if hasattr(entity, 'location'):
            return entity.location.shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    def entity(self, etype: Union[int, str], index: Optional[Index]=None):
        """
        Get entities in mesh structure.

        Parameters:
            etype (int | str): The topology dimension of the entity, or name
            index (int | slice | Field): The index of the entity.

        Returns:
            Entity: Entity or the default value.

        TODO:
            1. deal with index
        """
        edim = estr2dim(self, etype) if isinstance(etype, str) else etype
        if edim in self._entity_storage:
            entity = self._entity_storage[edim]
            if (index is None) or (index == _S) :
                return entity
            else:
                raise ValueError(f'Now we still can not deal with index which type is {type(index).__name__}.')
        else:
            return None

    def is_homogeneous(self) -> bool:
        """Return True if the mesh is homogeneous.

        Returns:
            bool: Homogeneous indiator.
        """
        return len(self.cell.shape) == 2

    ### topology
    def cell_to_node(self, return_sparse=True):
        edim = self.top_dimension()
        cell = self.entity(edim)
        if cell is None:
            raise ValueError(f'`cell` is `None`! There is no cell entity in mesh.')
        if return_sparse:
            NC = self.number_of_cells()
            return mesh_top_csr(cell, (NC, self.NN))
        else:
            return cell

    def face_to_node(self, return_sparse=True):
        edim = self.top_dimension() - 1
        face = self.entity(edim)
        if face is None:
            raise ValueError(f'`face` is `None`! There is no face entity in mesh.')
        if return_sparse:
            NF = self.number_of_faces()
            return mesh_top_csr(face, (NF, self.NN))
        else:
            return face 

    def edge_to_node(self, return_sparse=True):
        edge = self.entity(1)
        if edge is None:
            raise ValueError(f'`edge` is `None`! There is no edge entity in mesh.')
        if return_sparse:
            NE = self.number_of_faces()
            return mesh_top_csr(face, (NE, self.NN))
        else:
            return edge 

    def cell_to_edge(self, return_sparse=True):
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        cell2edge = self.cell2edge

        if return_sparse:
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            return mesh_top_csr(cell2edge, (NC, NE))
        else:
            return cell2edge

    def face_to_cell(self, return_sparse=False):
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell
        if return_sparse:
            NF = self.number_of_faces()
            NC = self.number_of_cells()
            entity = ti.field(face2cell.dtype, shape=(2*NF, ))
            location = ti.field(face2cell.dtype, shape=(NF+1, ))

            @ti.kernel
            def process_entity():
                for i, j in ti.ndrange(NF, 2):
                    entity[i*4 + j] = entity[i, j]
                for i in range(NF+1):
                    location[i] = i*2

            process_entity()
            return mesh_top_csr(face2cell, (NF, NC))
        else:
            return face2cell

    def boundary_face_flag(self) -> Field:
        """Return a boolean field indicating the boundary faces.

        Returns:
            Field: boundary face flag.
        """
        NF = self.number_of_faces()
        face2cell = self.face2cell

        bd_face_flag = ti.field(ti.u1, shape=(NF, ))

        @ti.kernel
        def fill_flag():
            for i in range(NF):
                bd_face_flag[i] = (face2cell[i, 0] == face2cell[i, 1])
        return bd_face_flag

    number_of_vertices_of_cells: _int_func = lambda self: self.cell.shape[-1]
    number_of_vertices_of_faces: _int_func = lambda self: self.localFace.shape[-1]
    number_of_vertices_of_edges: _int_func = lambda self: self.localEdge.shape[-1]
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_nodes_of_faces = number_of_vertices_of_faces
    number_of_nodes_of_edges = number_of_vertices_of_edges
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]

    def total_face(self) -> Field:
        """
        """
        NC = self.count('cell')
        NVF = self.number_of_nodes_of_faces()
        NFC = self.number_of_faces_of_cells()
        cell = self.entity(self.TD)
        local_face = self.localFace
        total_face = ti.field(self.itype, shape=(NFC*NC, NVF))   
        return total_face

    def total_edge(self) -> Field:
        """
        """
        NC = self.count('cell')
        NVE = self.number_of_nodes_of_edges()
        NEC = self.number_of_edges_of_cells()
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        total_edge = ti.field(self.itype, shape=(NEC*NC, NVE))   
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.cell.shape[0]
        NFC = self.cell.shape[1]

        totalFace = self.total_face()
        _, i0_np, j_np = np.unique(
            torch.sort(totalFace, dim=1)[0].cpu().numpy(),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = np.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = np.arange(NFC*NC, dtype=i0_np.dtype)

        self.cell2edge = torch.from_numpy(j_np).to(self.device).reshape(NC, NFC)
        self.cell2face = self.cell2edge

        face2cell_np = np.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = torch.from_numpy(face2cell_np).to(self.device)
        self.edge2cell = self.face2cell

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            total_edge = self.total_edge()
            _, i2, j = np.unique(
                torch.sort(total_edge, dim=1)[0].cpu().numpy(),
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = torch.from_numpy(j).to(self.device).reshape(NC, NEC)

        elif self.TD == 2:
            self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} edge (or face), "
                    f"on device {self.device}")
