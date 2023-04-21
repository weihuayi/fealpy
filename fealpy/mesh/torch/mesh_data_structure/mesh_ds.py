from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Optional, Union
from torch import Tensor

_VT = TypeVar('_VT')


class Redirector(Generic[_VT]):
    def __init__(self, target: str) -> None:
        self._target = target

    def __get__(self, obj, objtype) -> _VT:
        return getattr(obj, self._target)

    def __set__(self, obj, val: _VT):
        setattr(obj, self._target, val)


_int_redirectable = Union[int, Redirector[int]]
_tensor_redirectable = Union[Tensor, Redirector[Tensor]]


class MeshDataStructure(metaclass=ABCMeta):
    # Variables
    NN: int = -1
    cell: Tensor
    face: _tensor_redirectable
    edge: _tensor_redirectable
    edge2cell: Tensor

    # Constants
    TD: int
    localEdge: Tensor
    localFace: Tensor
    localCell: Tensor
    NVC: int
    NVE: _int_redirectable
    NVF: _int_redirectable
    NEC: _int_redirectable
    NFC: _int_redirectable

    def __init__(self, NN: int, cell: Tensor):
        self.itype = cell.dtype
        self.device = cell.device
        self.reinit(NN=NN, cell=cell)

    def reinit(self, NN: int, cell: Tensor):
        self.NN = NN
        self.cell = cell
        self.construct()

    @abstractmethod
    def construct(self) -> None:
        pass

    @property
    def number(self):
        return _Count(self) # Is this better?

    def number_of_cells(self):
        """Number of cells"""
        return self.cell.shape[0]

    def number_of_faces(self):
        """Number of faces"""
        return self.face.shape[0]

    def number_of_edges(self):
        """Number of edges"""
        return self.edge.shape[0]

    def number_of_nodes(self):
        """Number of nodes"""
        return self.NN

    def number_of_nodes_of_cells(self) -> int:
        """Number of nodes in a cell"""
        return self.cell.shape[-1]

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.NEC

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.NFC

    number_of_vertices_of_cells = number_of_nodes_of_cells


class _Count():
    def __init__(self, ds: MeshDataStructure) -> None:
        self._ds = ds

    def __call__(self, etype: Union[int, str]):
        TD = self._ds.TD
        if etype in {'cell', TD}:
            return self.nodes()
        elif etype in {'face', TD-1}:
            return self.faces()
        elif etype in {'edge', 1}:
            return self.edges()
        elif etype in {'node', 0}:
            return self.nodes()
        raise ValueError(f"Invalid entity type '{etype}'.")

    def nodes(self):
        return self._ds.NN

    def edges(self):
        return self._ds.edge.shape[0]

    def faces(self):
        return self._ds.face.shape[0]

    def cells(self):
        return self._ds.cell.shape[0]

    def nodes_of_cells(self):
        return self._ds.cell.shape[-1]
