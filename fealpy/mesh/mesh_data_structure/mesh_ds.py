from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Union

import numpy as np
from numpy.typing import NDArray

_VT = TypeVar('_VT')


class Redirector(Generic[_VT]):
    def __init__(self, target: str) -> None:
        self._target = target

    def __get__(self, obj, objtype) -> _VT:
        return getattr(obj, self._target)

    def __set__(self, obj, val: _VT) -> None:
        setattr(obj, self._target, val)


_int_redirectable = Union[int, Redirector[int]]
_array_redirectable = Union[NDArray, Redirector[NDArray]]


class MeshDataStructure(metaclass=ABCMeta):
    """
    @brief The abstract base class for all mesh types in FEALPy.

    This can not be instantialized before all abstract methods being implemented.\
    For now, only one method need to implement:
    - `construct(self) -> None`

    Besides, there are also some class attributes need to define:
    - `TD`: int, the topology dimension of mesh.
    - `NEC`: int, number of edges in each cell.
    - `NFC`: int, number of faces in each cell.

    This base class have already provide some frequently-used "counter" methods,
    such as `number_of_cells()`, `number_of_nodes_of_cells()` and other similar
    number-counting methods.

    This class is the very base class for mesh, containing some type annotation
    for common-used attributes.
    - For meshes with different topology dimension, we have subclasses
    `Mesh1dDataStructure`, `Mesh2dDataStructure` and `Mesh3dDataStructure` to
    tackle with the topology information of meshes with top dimension 1, 2 and 3
    respectively.
    - For structure and non-structure mesh, we have subclasses `Structure` and
    `Nonstructure` to control the initialization of a mesh.

    When define a data structure class for a particular type of meshes, use a
    proper combine of the subclasses mentioned above. For example, data structure
    class for triangle mesh can be defined as:
    ```
        class TriangleMeshDataStructure(Mesh2dDataStructure, Nonstructure):
            localEdge = ...
            localFace = ...
            ccw = ...
            NVC = 3
            NVE = 2
            NEC = 3
    ```
    Take 3-d uniform mesh as another example:
    ```
        class MyUniformMesh3dDS(Mesh3dDataStructure, Structure):
            localEdge = ...
            localFace = ...
            ccw = ...
    ```
    Usually, by subclassing these Mesh[x]dDatastructure and (Non)Structure, the
    class comes to final and can be used.

    A final mesh data structure class is supposed able to calculate any neighber
    relationship between mesh entities. These methods may be named like
    `cell_to_edge()`, `face_to_node()`, ...
    Some are even able to give the information of boundary, like
    `boundary_cell_flag()`.
    """
    # Variables
    itype: np.dtype
    NN: int = -1
    cell: NDArray
    face: _array_redirectable
    edge: _array_redirectable

    # Constants
    TD: int
    ccw: NDArray
    NVC: int
    NVE: _int_redirectable
    NVF: _int_redirectable
    NEC: _int_redirectable
    NFC: _int_redirectable

    @abstractmethod
    def construct(self) -> None:
        """
        @brief Construct critical data when initialized.
        """
        pass

    def clear(self) -> None:
        """
        @brief The inverse operator of `construct` method.
        """
        raise NotImplementedError

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

    def number_of_vertices_of_cells(self) -> int:
        """Number of vertices in a cell"""
        return self.cell.shape[-1]

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.NEC

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.NFC

    number_of_nodes_of_cells = number_of_vertices_of_cells
