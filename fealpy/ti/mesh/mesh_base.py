from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti

from .. import logger

EntityName = Literal['cell', 'cell_location', 'face', 'face_location', 'edge']
Field = TypeVar('Field') 
Index = Union[Field, int, slice]

_S = slice(None, None, None)
_T = TypeVar('_T')
_int_func = Callable[..., int]
_default = object()

def estr2dim(ds, etype: str) -> int:
    """
    """
    if etype == 'cell':
        return ds.top_dimension()
    elif etype == 'face':
        return ds.top_dimension() - 1
    elif etype == 'edge':
        return 1
    elif etype == 'node':
        return 0
    else:
        raise KeyError(f'{etype} is not a valid entity attribute.')


def edim2entity(ds, edim: int, index=None, *, default=_default):
    r"""Get entity field by its top dimension."""
    if edim in ds._entity_storage:
        et = ds._entity_storage[etype_dim]
        et.index = index
    else:
        if default is not _default:
            return default
        raise ValueError(f'{etype_dim} is not a valid entity attribute index '
                         f"in {ds.__class__.__name__}.")


def edim2node(ds, edim: int, index=None, dtype=None):
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    pass


class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge']
    def __init__(self, NN: int, TD: int) -> None: 
        self._entity_storage: Dict[int, _T] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> _T: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        edim = estr2dim(self, name)
        return edim2field(self, edim)

    def __setattr__(self, name: str, value: _T) -> None:
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
        if etype in ('node', 0):
            return self.NN
        edim = estr2dim(self, etype) if isinstance(etype, str) else etype

        entity = edim2entity(self, edim)
        if hasattr(entity, 'location'):
            return entity.location.shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.NN
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
            index (int | slice | Tensor): The index of the entity.
            etype (int | str): The topology dimension of the entity, or name
            default (Any): The default value if the entity is not found.

        Returns:
            Tensor: Entity or the default value.
        """

        if isinstance(etype, str):
            edim = estr2dim(self, etype)
        return edim2field(self, edim, index, default=default)

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
    def cell_to_node(self)
