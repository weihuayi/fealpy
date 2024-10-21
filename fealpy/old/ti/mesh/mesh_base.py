from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

from math import factorial, comb

import numpy as np
import taichi as ti

from .. import logger
from ..sparse import CSRMatrix
from .. import numpy as tnp

from .utils import EntityName, Entity, Field, Index, _S, _int_func
from .utils import estr2dim
from .quadrature import Quadrature
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
        entity = self.entity(etype) 
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
        Generate the total faces for every cells 

        Notes:
            tface = cell[:, lface].reshape(-1, NVF)
        """
        TD = self.TD
        NC = self.count('cell')
        NNF = self.number_of_nodes_of_faces()
        NFC = self.number_of_faces_of_cells()
        cell = self.entity(TD)
        lface = self.localFace
        tface = ti.field(self.itype, shape=(NFC*NC, NNF))   

        @ti.kernel
        def set_total_face():
            """
            Kernel function to set the total faces.
            """
            for i in range(NC):  # Iterate over each cell
                for j in range(NFC):  # Iterate over each face in the cell
                    for k in range(NVF):  # Iterate over each vertex in the face
                        tface[i * NFC + j, k] = cell[i, lface[j, k]]  # Set total face value

        set_total_face()  # Call the kernel function to set total faces
        return tface

    def total_edge(self) -> Field:
        """
        Generate the total edges fro every cells 
        """
        TD = self.TD
        NC = self.count('cell')
        NNE = self.number_of_nodes_of_edges()
        NEC = self.number_of_edges_of_cells()
        cell = self.entity(TD)
        ledge = self.localEdge
        tedge = ti.field(self.itype, shape=(NEC*NC, NNE))   

        @ti.kernel
        def set_total_edge():
            """
            Kernel function to set the total faces.
            """
            for i in range(NC):  
                for j in range(NEC):
                    for k in range(NNE):
                        tface[i * NFC + j, k] = cell[i, ledge[j, k]]
        return tedge

    def construct(self):
        """
        Consruct the adjacency relationship

        Notes:
            Here we use numpy, and will update it to taichi in future.
        """
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.count('cell')
        NNF = self.number_of_nodes_of_faces()
        NFC = self.number_of_faces_of_cells()

        cell = self.entity('cell').to_numpy()
        lface = self.localFace.to_numpy()
        tface = cell[:, lface].reshape(-1, NNF)
        _, i0, j = np.unique(
            np.sort(tface, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        NF = i0.shape[0]
        face = tface[i0, :]

        face2cell = np.zeros((NF, 4), dtype=np.int32)
        i1 = np.zeros(NF, dtype=np.int32)
        i1[j] = np.arange(NFC*NC, dtype=np.int32)

        face2cell[:, 0] = i0 // NFC
        face2cell[:, 1] = i1 // NFC
        face2cell[:, 2] = i0 % NFC
        face2cell[:, 3] = i1 % NFC

        cell2face = j.reshape(-1, NFC)

        self.face = tnp.from_numpy(face)
        self.face2cell = tnp.from_numpy(face2cell)
        self.cell2face = tnp.from_numpy(cell2face)

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()
            NNE = self.number_of_nodes_of_edges()
            ledge = self.localEdge.to_numpy()
            tedge = cell[:, ledge].reshape(-1, NNE) 
            _, i2, j = np.unique(
                np.sort(tedge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0
            )
            edge = tedge[i2, :]
            cell2edge = np.reshape(j, (NC, NEC))
            self.edge = tnp.from_numpy(edge)
            self.cell2edge = tnp.from_numpy(cell2edge)

        elif self.TD == 2:
            self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} edge (or face), ")


##################################################
### Mesh
##################################################

class Mesh(MeshDS):
    @property
    def ftype(self):
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the float type as the node '
                               'has not been assigned.')
        return node.dtype

    def geo_dimension(self) -> int:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]

    GD = property(geo_dimension)

    def multi_index_matrix(self, p: int, edim: int) -> Field:
        return F.multi_index_matrix(p, edim)

    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Field:
        """Get the barycenter of the entity.

        Parameters:
            etype (int | str): The topology dimension of the entity, or name
                'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
            index (int | slice | Field): The index of the entity.

        Returns:
            Field: A 2-d scalar field containing barycenters of the entity.
        """

        assert index is None, "Up to now, we just support the case index==None"

        if etype in ('node', 0):
            return self.node
        node = self.entity(0)
        entity = self.entity(etype)
        return F.entity_barycenter(entity, node)

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        logger.warning("The `integrator` is deprecated and will be removed after 3.0. "
                       "Use `quadrature_formula` instead.")
        return self.quadrature_formula(q, etype, qtype)


class HomogeneousMesh(Mesh):
    def interpolation_points(self, p: int, index: Index=_S) -> Field:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Field:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> Field:
        raise NotImplementedError

class SimplexMesh(HomogeneousMesh):
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            dim = estr2dim(self, iptype)
        return F.simplex_ldof(p, dim)

    def number_of_global_ipoints(self, p: int):
        return F.simplex_gdof(p, self)

    def grad_lambda(self, index: Index=_S) -> Field:
        raise NotImplementedError

    def shape_function(self, bc: Field, p: int=1, *, 
                       variable: str='u', mi: Optional[Field]=None) -> Field:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD)
        phi = F.simplex_shape_function(bc, p, mi)
        if variable == 'u':
            return phi
        elif variable == 'x':
            return phi.unsqueeze_(1)
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")
