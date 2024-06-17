from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
from scipy.sparse import csr_matrix

from .. import logger

from .utils import EntityName, Entity, Index, _T, _S, _int_func
from .utils import estr2dim, arr_to_csr
from . import functional as F


class MeshDS():
    ccw: np.ndarray 
    localEdge: np.ndarray 
    localFace: np.ndarray 
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
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
        if isinstance(entity, dict):
            return entity['location'].shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    def entity(self, etype: Union[int, str], index: Optional[Index]=_S):
        """
        Get entities in mesh data structure.

        Parameters:
            etype (int | str): The topology dimension of the entity, or name
            index (int | slice | np.ndarray): The index of the entity.
        Returns:
            entity (np.ndarray | list):
        """

        edim = estr2dim(self, etype) if isinstance(etype, str) else etype
        if edim in ds._entity_storage:
            entity = ds._entity_storage[edim]
            if isinstance(entity, dict): 
                entity = np.hsplit(entity['entity'], entity['location'][1:-1])
            return entity[index]
        else:
            return None

    def is_homogeneous(self) -> bool:
        """Return True if the mesh is homogeneous.

        Returns:
            bool: Homogeneous indiator.
        """
        return isinstance(entity, dict) 

    ### topology
    def cell_to_node(self, return_sparse: bool=True, dtype=np.bool_):
        edim = self.top_dimension()
        NN = self.count('node')
        cell = self.entity(edim) 
        if return_sparse:
            return arr_to_csr(cell, NN, dtype=dtype)
        else:
            return cell 

    def face_to_node(self, return_sparse: bool=True, dtype=np.bool_):
        edim = self.top_dimension() - 1
        NN = self.count('node')
        face = self.entity(edim)
        if return_sparse:
            return arr_to_csr(face, NN, dtype=np.bool_)
        else:
            return face 

    def edge_to_node(self, return_sparse: bool=True, dtype=np.bool_):
        NN = self.count('node')
        edge = self.entity(1)
        if return_sparse:
            return arr_to_csr(edge, NN, dtype=np.bool_)
        else:
            return edge 

    def cell_to_edge(self, return_sparse: bool=False, dtype=np.bool_):
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        cell2edge = self.cell2edge
        NE = self.count('edge')
        if return_sparse:
            return arr_to_csr(cell2edge, NE, dtype=dtype)
        else:
            return cell2edge

    def face_to_cell(self, return_sparse: bool=False, dtype=np.bool_):
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell
        NC = self.count('cell')
        if return_sparse:
            return arr_to_csr(face2cell[:, :2], NC, dtype=dtype)
        else:
            return face2cell

    def edge_to_cell(self, return_sparse: bool=False, dtype=np.bool_):
        """
        TODO:
            1. Need to update code for 3D case
        """
        assert self.TD == 2, "Now we just suport mesh which TD==2"
        if not hasattr(self, 'edge2cell'):
            raise RuntimeError('Please call construct() first or make sure the edge2cell'
                               'has been constructed.')
        edge2cell = self.edge2cell
        NC = self.count('cell')
        if return_sparse:
            return arr_to_csr(edge2cell[:, :2], NC, dtype=dtype)
        else:
            return edge2cell

    ### boundary
    def boundary_node_flag(self) -> np.ndarray:
        """
        Return a boolean array indicating the boundary nodes.

        Returns:
            np.ndarray : boundary node flag.
        """
        pass

    def boundary_face_flag(self) -> Tensor:
        """
        Return a boolean array indicating the boundary faces.

        Returns:
            Tensor: boundary face flag.
        """
        return self.face2cell[:, 0] == self.face2cell[:, 1]

    def boundary_cell_flag(self) -> np.ndarray:
        """Return a boolean tensor indicating the boundary cells.

        Returns:
            np.ndarray : boundary cell flag.
        """
        raise NotImplementedError

    def boundary_node_index(self): 
        raise NotImplementedError

    def boundary_face_index(self): 
        raise NotImplementedError

    def boundary_cell_index(self): 
        raise NotImplementedError

    number_of_vertices_of_cells: _int_func = lambda self: self.cell.shape[-1]
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]
    number_of_vertices_of_faces: _int_func = lambda self: self.localFace.shape[-1]
    number_of_vertices_of_edges: _int_func = lambda self: self.localEdge.shape[-1]

    def total_face(self) -> np.ndarray:
        NVF = self.number_of_vertices_of_faces()
        cell = self.entity(self.TD)
        total_face = cell[..., self.local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> np.ndarray:
        NVE = self.number_of_vertices_of_edges()
        cell = self.entity(self.TD)
        total_edge = cell[..., self.local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.number_of_cells()

        total_face = self.total_face()
        _, i0, j = np.unique(
            np.sort(total_face, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = total_face[i0, :]
        NFC = self.number_of_faces_of_cells()
        NF = i0.shape[0]

        self.face2cell = np.zeros((NF, 4), dtype=self.itype)

        i1 = np.zeros(NF, dtype=self.itype)
        i1[j] = np.arange(NFC*NC, dtype=self.itype)

        self.face2cell[:, 0] = i0 // NFC
        self.face2cell[:, 1] = i1 // NFC
        self.face2cell[:, 2] = i0 % NFC
        self.face2cell[:, 3] = i1 % NFC

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()
            total_edge = self.total_edge()

            _, i2, j = np.unique(
                np.sort(total_edge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = np.reshape(j, (NC, NEC)) # 原来是 NFC, 应为 NEC

        elif self.TD == 2:
            self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} edge (or face), ")


class Mesh(MeshDS):
    @property
    def ftype(self) -> _dtype: return self.node.dtype

    def geo_dimension(self) -> int: return self.node.shape[-1]
    GD = property(geo_dimension)

    def multi_index_matrix(self, p: int, etype: int) -> Tensor:
        raise NotImplementedError

    def entity_barycenter(self, 
            etype: Union[int, str], 
            index: Optional[Index]=None) -> np.ndarray: 
        """
        Get the barycenter of the entity.

        Args:
            etype (int | str): The topology dimension of the entity, or name
            'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: A 2-d tensor containing barycenters of the entity.
        """
        node = self.entity('node')
        entity = self.entity(etype, index=index) 
        return F.entity_barycenter(entity, node)

    def edge_length(self, index: Index=_S, out=None) -> Tensor:
        """Calculate the length of the edges.

        Args:
            index (int | slice | Tensor, optional): Index of edges.
            out (Tensor, optional): The output tensor. Defaults to None.

        Returns:
            Tensor: Length of edges, shaped [NE,].
        """
        edge = self.entity(1, index=index)
        return F.edge_length(self.node[edge], out=out)

    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> Tensor:
        """
        Calculate the normal of the edges.

        Args:
            index (int | slice | Tensor, optional): Index of edges.
            unit (bool, optional): _description_. Defaults to False.
            out (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        edge = self.entity(1, index=index)
        return F.edge_normal(self.node[edge], unit=unit, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> Tensor:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.
        """
        return self.edge_normal(index=index, unit=True, out=out)

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        """Get the quadrature points and weights."""
        raise NotImplementedError

    def shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        """Shape function value on the given bc points, in shape (..., ldof).

        Args:
            bc (Tensor): The bc points, in shape (..., NVC).
            p (int, optional): The order of the shape function. Defaults to 1.
            index (int | slice | Tensor, optional): The index of the cell.
            variable (str, optional): The variable name. Defaults to 'u'.
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (..., ldof). The shape will\
            be (..., 1, ldof) if `variable == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")


class HomogeneousMesh(Mesh):
    pass


class SimplexMesh(HomogeneousMesh):
    pass
