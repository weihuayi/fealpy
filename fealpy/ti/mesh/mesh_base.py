from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti

from ..sparse import CSRMatrix
from .. import logger


EntityName = Literal['cell', 'cell_location', 'face', 'face_location', 'edge']
Entity = TypeVar('Entity') 
Field = TypeVar('Field')
Index = Union[Field, int, slice]

_S = slice(None, None, None)
_T = TypeVar('_T')
_int_func = Callable[..., int]
_default = object()

def mesh_top_csr(entity: Entity, shape: Tuple(int, int), copy=False):
    if entity.ndim == 1:
        if (~hasattr(entity, 'location')) or (entity.location is None):
             raise ValueError('entity.location is required for 1D entity (usually for polygon mesh).')
         indices = entity
         indptr = entity.location
    elif entity.ndim == 2: # for homogeneous case
        M = entity.shape[0]
        N = entity.shape[1]
        indices = ti.field(entity.dtype, shape=(M*N, ))
        indptr = ti.field(entity.dtype, shape=(M+1, ))

        @ti.kernel
        def process_entity():
            for i, j in ti.ndrange(M, N):
                indices[i*N + j] = entity[i, j]
            for i in range(M+1):
                indptr[i] = i*N

        process_entity()
    else:
        raise ValueError('dimension of entity must be 1 or 2.')
    # Notice that the data can be None for CSRMatrix, which intend to save memory
    return CSRMatrix((None, entity, entity.location), shape=shape, copy=copy) 

def estr2dim(ds, estr: str) -> int:
    """
    """
    if estr == 'cell':
        return ds.top_dimension()
    elif estr == 'face':
        return ds.top_dimension() - 1
    elif estr == 'edge':
        return 1
    elif estr == 'node':
        return 0
    else:
        raise KeyError(f'{estr} is not a valid entity attribute.')


def edim2entity(ds, edim: int, index=None, *, default=_default):
    r"""Get entity by its top dimension."""
    if edim in ds._entity_storage:
        entity = ds._entity_storage[edim]
        if index is None:
            return entity
        else:
            raise ValueError(f'Now we just can deal with index=None case!')
    else:
        return None


def edim2node(ds, edim: int, index=None, dtype=None):
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    pass


class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    def __init__(self, NN: int, TD: int) -> None: 
        self._entity_storage: Dict[int, _T] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> _T: ...
    def __getattr__(self, name: str):
        """
        """
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        edim = estr2dim(self, name)
        return edim2entity(self, edim)

    def __setattr__(self, name: str, value: Entity) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('Please call super().__init__() before setting attributes!')
            edim = estr2dim(self, name)
            self._entity_storage[edim] = value
        else:
            super().__setattr__(name, value) # object()


    ### properties
    def top_dimension(self) -> int: return self.TD

    @property
    def itype(self): return self.cell.dtype

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        edim = estr2dim(self, etype) if isinstance(etype, str) else etype
        entity = edim2entity(self, edim)
        if hasattr(entity, 'location'):
            return entity.location.shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None): ...
    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default: _T): ...
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default=_default):
        """Get entities in mesh structure.

        Args:
            etype (int | str): The topology dimension of the entity, or name
            index (int | slice | Field): The index of the entity.
            default (Any): The default value if the entity is not found.

        Returns:
            Entity: Entity or the default value.
        """
        edim = estr2dim(self, etype) if isinstance(etype, str) else etype
        return edim2entity(self, edim, index, default=default)

    def total_face(self) -> Template:
        raise NotImplementedError

    def total_edge(self) -> Tensor:
        raise NotImplementedError

    def is_homogeneous(self) -> bool:
        """Return True if the mesh is homogeneous.

        Returns:
            bool: Homogeneous indiator.
        """
        return len(self.cell.shape) == 2

    ### topology
    def cell_to_node(self):
        TD = self.top_dimension()
        return entity_dim2node(self, TD, index, dtype=dtype)

    def face_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None):
        etype = self.top_dimension() - 1
        return entity_dim2node(self, etype, index, dtype=dtype)

    def edge_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None):
        return entity_dim2node(self, 1, index, dtype)

    def cell_to_node(self)
