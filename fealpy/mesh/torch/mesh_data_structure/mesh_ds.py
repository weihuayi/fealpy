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
    ccw: Tensor

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
        self.reinit(NN=NN, cell=cell)

    def reinit(self, NN: int, cell: Tensor):
        self.NN = NN
        self.cell = cell
        self.itype = cell.dtype
        self.device = cell.device
        self.construct()

    @abstractmethod
    def construct(self) -> None:
        """
        @brief Construct the topology data structure.

        This is called automatically in initialization, and there are no need\
        for users to call this.
        """
        pass

    def clear(self) -> None:
        raise NotImplementedError

    def number_of_cells(self) -> int:
        """Number of cells"""
        return self.cell.shape[0]

    def number_of_faces(self) -> int:
        """Number of faces"""
        return self.face.shape[0]

    def number_of_edges(self) -> int:
        """Number of edges"""
        return self.edge.shape[0]

    def number_of_nodes(self) -> int:
        """Number of nodes"""
        return self.NN

    def number_of_vertices_of_cells(self) -> int:
        """Number of nodes in a cell"""
        return self.cell.shape[-1]

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.NEC

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.NFC

    number_of_nodes_of_cells = number_of_vertices_of_cells
