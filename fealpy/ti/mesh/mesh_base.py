from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti
from ti.types import template as Template 
from ti.types import ndarray as NDArray

from .. import logger
from .utils import EntityName, entity_str2dim, entity_dim2field, _T, _default

Index = Union[Template, int, slice]

class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'cell_location', 'face_location']
    def __init__(self, NN: int, TD: int) -> None: 
        self._entity_storage: Dict[int, Template] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Template: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        edim = entity_str2dim(self, name)
        return entity_dim2field(self, edim)

    def __setattr__(self, name: str, value: Template) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('Please call super().__init__() before setting attributes!')
            edim = entity_str2dim(self, name)
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
        if isinstance(etype, str):
            edim = entity_str2dim(self, etype)
        if -edim in self._entity_storage: # for polygon mesh
            return self._entity_storage[-edim].size(0) - 1
        return entity_dim2field(self, edim).shape[0]

    def number_of_nodes(self): return self.NN
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Template: ...
    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default: _T) -> Union[Template, _T]: ...
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default=_default):
        if isinstance(etype, str):
            etype = entity_str2dim(self, etype)
        return entity_dim2field(self, etype, index, default=default)

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
