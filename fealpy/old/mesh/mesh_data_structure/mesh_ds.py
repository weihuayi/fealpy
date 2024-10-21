from typing import TypeVar, Generic, Union, Callable, overload

import numpy as np
from numpy import dtype
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .sparse_tool import enable_csr, arr_to_csr

_VT = TypeVar('_VT')


class Redirector(Generic[_VT]):
    def __init__(self, target: str) -> None:
        self._target = target

    def __get__(self, obj, objtype) -> _VT:
        return getattr(obj, self._target)

    def __set__(self, obj, val: _VT) -> None:
        setattr(obj, self._target, val)

    def __delete__(self, obj) -> None:
        delattr(obj, self._target)


ArrRedirector = Redirector[NDArray]
_array_redirectable = Union[NDArray, Redirector[NDArray]]


class MeshDataStructure():
    """
    @brief The abstract base class for all mesh data structure types in FEALPy.

    This can not be instantialized before all abstract methods being implemented.

    Besides, this class attribute need to define:
    - `TD`: int, the topology dimension of mesh.

    This base class have already provide some methods:
    - Number of entities:
    such as `number_of_cells()`, `number_of_nodes_of_cells()` and other similar
    number-counting methods.
    - Neighbor info from other to node:
    they are `cell_to_node`, `face_to_node`, `edge_to_node`.

    A final mesh data structure class is supposed able to calculate critical
    relationship between mesh entities. They are abstracts listed below.

    Abstract methods list:
    - `cell_to_edge`
    - `cell_to_face`
    - `face_to_cell`
    """
    # Variables
    itype: np.dtype
    NN: int = -1
    cell: _array_redirectable
    face: _array_redirectable
    edge: _array_redirectable

    # Constants
    TD: int

    # counters

    def number_of_cells(self):
        """
        @brief Return the number of cells in the mesh.

        This is done by getting the length of `cell` array.
        """
        return len(self.cell)

    def number_of_faces(self):
        """
        @brief Return the number of faces in the mesh.

        This is done by getting the length of `face` array. Make sure the `face`
        array is calculated when calling this method.
        """
        return len(self.face)

    def number_of_edges(self):
        """
        @brief Return the number of edges in the mesh.

        This is done by getting the length of `edge` array. Make sure the `edge`
        array is calculated when calling this method.
        """
        return len(self.edge)

    def number_of_nodes(self):
        """
        @brief Return the number of nodes in the mesh.
        """
        return self.NN

    # topology

    def cell_to_node(self, *args, **kwargs) -> NDArray:
        """
        @brief Return neighbor information from cell to node.
        """
        raise NotImplementedError

    def cell_to_edge(self, *args, **kwargs) -> NDArray:
        raise NotImplementedError

    def cell_to_face(self, *args, **kwargs) -> NDArray:
        raise NotImplementedError

    def face_to_cell(self, *args, **kwargs) -> NDArray:
        raise NotImplementedError

    def face_to_node(self) -> NDArray:
        raise NotImplementedError

    def edge_to_node(self, return_sparse=False, return_local=False):
        if not return_sparse:
            return self.edge
        else:
            return arr_to_csr(self.edge, self.number_of_nodes(),
                              return_local=return_local, dtype=self.itype)

    def node_to_edge(self, return_local=False):
        return arr_to_csr(self.edge, self.number_of_nodes(),
                          reversed=True, return_local=return_local, dtype=self.itype)

    def node_to_node(self):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edge = self.edge
        node2node = csr_matrix(
                (
                    np.ones((2*NE,), dtype=np.bool_),
                    (
                        edge.flat,
                        edge[:, [1, 0]].flat
                    )
                ), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node * edge2node.T


    # boundary flag

    def boundary_node_flag(self) -> NDArray:
        """
        @brief Return a bool array to show whether nodes are on the boundary.
        """
        NN = self.number_of_nodes()
        face2node = self.face_to_node()
        is_bd_face = self.boundary_face_flag()
        is_bd_node = np.zeros((NN, ), dtype=np.bool_)
        is_bd_node[face2node[is_bd_face, :]] = True
        return is_bd_node

    def boundary_edge_flag(self) -> NDArray:
        """
        @brief Return a bool array to show whether edges are on the boundary of\
               a 3-d mesh.

        @note: For 2-d meshes, `boundary_edge_flag` should be assigned to `boundary_face_flag`.
        """
        NE = self.number_of_edges()
        face_to_edge_fn = getattr(self, 'face_to_edge', None)
        if face_to_edge_fn is None:
            raise NotImplementedError(f"The neighbor info method 'face_to_edge()'\
                                      should be implemented for finding boundary edges.")
        face2edge = face_to_edge_fn()
        is_bd_face = self.boundary_face_flag()
        is_bd_edge = np.zeros((NE,), dtype=np.bool_)
        is_bd_edge[face2edge[is_bd_face, :]] = True
        return is_bd_edge

    def boundary_face_flag(self) -> NDArray:
        """
        @brief Return a bool array to show whether faces are on the boundary.
        """
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self) -> NDArray:
        """
        @brief Return a bool array to show whether cells are next to the boundary.
        """
        NC = self.number_of_cells()
        face2cell = self.face_to_cell()
        is_bd_face = self.boundary_face_flag()
        is_bd_cell = np.zeros((NC, ), dtype=np.bool_)
        is_bd_cell[face2cell[is_bd_face, 0]] = True
        return is_bd_cell

    # boundary index

    def boundary_node_index(self) -> NDArray:
        """
        @brief Find the indexes of nodes on the boundary.
        """
        isBdNode = self.boundary_node_flag()
        idx, = np.nonzero(isBdNode)
        return idx

    def boundary_edge_index(self) -> NDArray:
        """
        @brief Find the indexes of edges on the boundary.
        """
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_face_index(self) -> NDArray:
        """
        @brief Find the indexes of faces on the boundary.
        """
        isBdFace = self.boundary_face_flag()
        idx, = np.nonzero(isBdFace)
        return idx

    def boundary_cell_index(self) -> NDArray:
        """
        @brief Find the indexes of cells next to the boundary.
        """
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx

    # boundary entity

    def boundary_edge(self) -> NDArray:
        return self.edge[self.boundary_edge_flag()]

    def boundary_face(self) -> NDArray:
        return self.face[self.boundary_face_flag()]

    def boundary_cell(self) -> NDArray:
        return self.cell[self.boundary_cell_flag()]


class HomogeneousMeshDS(MeshDataStructure):
    """
    @brief Data structure for meshes with homogeneous shape of cells.

    This subclass is to implement:
    - Construction of `face2cell` and `face`, basically;
    - Basic topology relationship: `face2cell` and `cell2edge`(in 3d case);
    - Special counting methods, calculating NVC, NEC, NFC, NVF, and NVE;
    - Homogeneous local entities, like `local_edge` and `local_face`;
    - Generate total entities: `total_edge` and `total_face`;
    - Topology relationship between cell, face and node; the methods are for all\
    dimensions, but may have simpler algorithms in low dimension, which can be\
    implemented overridingly. Note: Relations between `edge` and entities with\
    dimension higher than it (face and cell) are different in 1d, 2d and 3d.

    Class variables:
    ccw: NDArray, optional. The indices of nodes sorted counter-clock-wise in\
         a face(3d case) or a cell(2d case). This will be checked when plotting\
         the mesh. If not provided, the original order of nodes will be used.
    localEdge: NDArray with shape (NEC, NVE).
    localFace: NDArray with shape (NFC, NVF).

    @note `face2cell` and `face` will be construct for all homogeneous meshes, but
    there are some differences in 1d, 2d and 3d case.
    - 1d: Redirect `edge` to `cell`, and the `face` is like [[0], [1], [2], ..., [NN-1]];
    - 2d: Redirect `edge` to `face`;
    - 3d: Also construct `cell2edge` and `edge`.
    """
    # Constants
    ccw: NDArray
    localEdge: NDArray
    localFace: NDArray

    def __init__(self, NN: int, cell: NDArray) -> None:
        self.reinit(NN=NN, cell=cell)

    def reinit(self, NN: int, cell: NDArray):
        if not isinstance(NN, int):
            raise TypeError("Expect int for number of nodes, "
                            f"but got {NN.__class__.__name__}.")
        if NN <= 0:
            raise ValueError("The number of nodes must be positive, "
                             f"but got {NN}.")
        self.NN = NN

        if not isinstance(cell, np.ndarray):
            raise TypeError("Expect numpy array for cell entity, "
                            f"but got {cell.__class__.__name__}.")
        if cell.ndim != 2:
            raise ValueError("Cell array should be 2-dimensional, "
                             f"but got array with shape {cell.shape}.")
        self.cell = cell
        self.itype = cell.dtype
        self.construct()

    def construct(self) -> None:
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

    def clean(self) -> None:
        del self.face # this also deletes edge in 2-d mesh.
        del self.face2cell

        if self.TD == 3:
            del self.edge
            del self.cell2edge
        elif self.TD == 2:
            del self.edge2cell

    def number_of_vertices_of_cells(self) -> int:
        """
        @brief Return the number of vertices in a cell.
        """
        if hasattr(self, 'NVC'):
            return getattr(self, 'NVC')
        return self.cell.shape[-1]

    def number_of_edges_of_cells(self) -> int:
        """
        @brief Return the number of edges in a cell.

        This is equal to the length of `localEdge` in axis-0, usually be marked
        as NEC.
        """
        if hasattr(self, 'NEC'):
            return getattr(self, 'NEC')
        return self.localEdge.shape[0]

    def number_of_faces_of_cells(self) -> int:
        """
        @brief Return the number of faces in a cell.

        This is equal to the length of `localFace` in axis-0, usually be marked
        as NFC.
        """
        if hasattr(self, 'NFC'):
            return getattr(self, 'NFC')
        return self.localFace.shape[0]

    def number_of_vertices_of_faces(self) -> int:
        """
        @brief Return the number of vertices in a face.

        This is equal to the length of `localFace` in axis-1, usually be marked
        as NVF.
        """
        if hasattr(self, 'NVF'):
            return getattr(self, 'NVF')
        return self.localFace.shape[-1]

    def number_of_vertices_of_edges(self) -> int:
        """
        @brief Return the number of vertices in an edge.

        This is equal to the length of `localEdge` in axis-1, usually be marked
        as NVE.
        """
        if hasattr(self, 'NVE'):
            return getattr(self, 'NVE')
        return self.localEdge.shape[-1]

    number_of_nodes_of_cells = number_of_vertices_of_cells

    @classmethod
    def local_face(cls):
        return cls.localFace

    @classmethod
    def local_edge(cls):
        return cls.localEdge

    def total_face(self) -> NDArray:
        NVF = self.number_of_vertices_of_faces()
        cell = self.cell
        local_face = self.localFace
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> NDArray:
        NVE = self.number_of_vertices_of_edges()
        cell = self.cell
        local_edge = self.localEdge
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    # between (cell, face) and node

    def cell_to_node(self, return_sparse=False, return_local=False):
        if not return_sparse:
            return self.cell
        else:
            return arr_to_csr(self.cell, self.NN,
                              return_local=return_local, dtype=self.itype)

    def node_to_cell(self, return_local=False):
        return arr_to_csr(self.cell, self.NN, reversed=True,
                          return_local=return_local, dtype=self.itype)

    def face_to_node(self, return_sparse=False, return_local=False):
        if not return_sparse:
            return self.face
        else:
            return arr_to_csr(self.face, self.NN,
                              return_local=return_local, dtype=self.itype)

    def node_to_face(self, return_local=False):
        return arr_to_csr(self.face, self.NN, reversed=True,
                          return_local=return_local, dtype=self.itype)

    # between cell and face

    def cell_to_face(self, return_sparse=False, return_local=False) -> NDArray:
        """
        @brief Neighbor information of cell to face.
        """
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        NFC = self.number_of_faces_of_cells()

        face2cell = self.face2cell
        cell2face = np.zeros((NC, NFC), dtype=self.itype)
        cell2face[face2cell[:, 0], face2cell[:, 2]] = range(NF)
        cell2face[face2cell[:, 1], face2cell[:, 3]] = range(NF)
        if not return_sparse:
            return cell2face
        else:
            return arr_to_csr(cell2face, self.number_of_faces(),
                              return_local=return_local, dtype=self.itype)

    def face_to_cell(self, return_sparse=False): # TODO: return local in sparse
        if return_sparse is False:
            return self.face2cell
        else:
            return arr_to_csr(
                self.face2cell[:, [0, 1]], self.number_of_cells())

    def cell_to_cell(self, return_sparse=False,
                     return_boundary=True, return_array=False):
        if return_array:
            return_sparse = False
            return_boundary = False

        NC = self.number_of_cells()
        NF = self.number_of_faces()

        face2cell = self.face2cell
        if (return_sparse is False) and (return_array is False):
            NFC = self.number_of_faces_of_cells()
            cell2cell = np.zeros((NC, NFC), dtype=self.itype)
            cell2cell[face2cell[:, 0], face2cell[:, 2]] = face2cell[:, 1]
            cell2cell[face2cell[:, 1], face2cell[:, 3]] = face2cell[:, 0]
            return cell2cell

        val = np.ones((NF, ), dtype=np.bool_)

        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (face2cell[:, 0], face2cell[:, 1])),
                    shape=(NC, NC))
            cell2cell += coo_matrix(
                    (val, (face2cell[:, 1], face2cell[:, 0])),
                    shape=(NC, NC))
            return cell2cell.tocsr()

        isInFace = (face2cell[:, 0] != face2cell[:, 1])
        cell2cell = coo_matrix(
                (
                    val[isInFace],
                    (face2cell[isInFace, 0], face2cell[isInFace, 1])
                ),
                shape=(NC, NC), dtype=np.bool_)
        cell2cell += coo_matrix(
                (
                    val[isInFace],
                    (face2cell[isInFace, 1], face2cell[isInFace, 0])
                ), shape=(NC, NC), dtype=np.bool_)
        cell2cell = cell2cell.tocsr()

        if not return_array:
            return cell2cell

        nn = cell2cell.sum(axis=1).reshape(-1)
        _, adj = cell2cell.nonzero()
        adjLocation = np.zeros(NC+1, dtype=np.int32)
        adjLocation[1:] = np.cumsum(nn)
        return adj.astype(np.int32), adjLocation


class StructureMeshDS(HomogeneousMeshDS):
    """
    @brief Base class of data structure for structure meshes.

    Subclass to change nonstructure mesh type to structure mesh type.
    """
    # Variables
    cell = ArrRedirector('cell_')

    # Constants
    TD: int

    def __init__(self, *nx: int, itype: dtype) -> None:
        if len(nx) != self.TD:
            raise ValueError(f"Number of `nx` must match the top dimension.")
        for nx_item in nx:
            if not isinstance(nx_item, int):
                raise TypeError(f"Expect int for nx, but got {nx_item}.")

        self.nx_ = np.array(nx, dtype=itype)
        self.NN = np.prod(self.nx_ + 1)
        self.itype = itype

    @property
    def nx(self):
        return self.nx_[0]
    @property
    def ny(self):
        return self.nx_[1]
    @property
    def nz(self):
        return self.nx_[2]

    def number_of_cells(self) -> int:
        """
        @brief Return the number of cells in the struct mesh.

        This is equal to nx*ny*nz*...
        """
        return np.prod(self.nx_, axis=0)

    def number_of_faces(self):
        """
        @brief Return the number of faces in the struct mesh.
        """
        full = np.prod(self.nx_, axis=0)
        adds = full // self.nx_
        return full*self.TD + np.sum(adds, axis=0)

    def number_of_edges(self):
        """
        @brief Return the number of edges in the struct mesh.
        """
        full = np.prod(self.nx_ + 1, axis=0)
        subs = full // (self.nx_ + 1)
        return full*self.TD - np.sum(subs, axis=0)

    @property
    def cell_(self):
        TD = self.TD
        NN = self.NN
        NC = np.prod(self.nx_)
        cell = np.zeros((NC, 2**TD), dtype=self.itype)
        idx = np.arange(NN).reshape(self.nx_+1)
        c = idx[(slice(-1), )*TD]
        cell[:, 0] = c.flat

        ## This is for any topology dimension:

        # for i in range(1, TD + 1):
        #     begin = 2**(i-1)
        #     end = 2**i
        #     jump = np.prod(self._nx+1)//(self.nx+1)
        #     cell[:, begin:end] = cell[:, 0:end-begin] + jump

        if TD >= 1:
            cell[:, 1:2] = cell[:, 0:1] + 1

        if TD >= 2:
            cell[:, 2:4] = cell[:, 0:2] + self.ny + 1

        if TD >= 3:
            cell[:, 4:8] = cell[:, 0:4] + (self.ny+1)*(self.nz+1)

        return cell

    def construct(self) -> None:
        """
        @brief Warning: `construct` method is not available any more in structure\
               meshes. This raises NotImplementedError when called.
        """
        raise NotImplementedError("'construct' method is unnecessary for"
                                  "structure meshes.")
