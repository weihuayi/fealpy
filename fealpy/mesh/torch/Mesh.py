
from typing import Optional, Generic, TypeVar, Union
from torch import Tensor
import numpy as np


_VT = TypeVar('_VT')

class Redirector(Generic[_VT]):
    def __init__(self, target: str) -> None:
        self._target = target

    def __get__(self, obj, objtype) -> _VT:
        return getattr(obj, self._target)

    def __set__(self, obj, val: _VT):
        setattr(obj, self._target, val)


class MeshDataStructure():
    # Variables
    NN: int = -1
    cell: Tensor
    face: Optional[Tensor]
    edge: Optional[Tensor]
    edge2cell: Optional[Tensor]

    # Constants
    TD: int
    localEdge: Tensor
    localFace: Tensor
    localCell: Tensor
    NVC: int
    NVE: int
    NVF: int
    NEC: int
    NFC: int

    def __init__(self, NN: int, cell: Tensor):
        self.itype = cell.dtype
        self.device = cell.device
        self.reinit(NN=NN, cell=cell)

    def reinit(self, NN: int, cell: Tensor):
        self.NN = NN
        self.cell = cell
        self.construct()

    def construct(self):
        raise NotImplementedError

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


class Mesh():
    ds: MeshDataStructure
    node: Tensor

    ### General Interfaces ###

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.number_of_edges()

    def number_of_faces(self) -> int:
        return self.ds.number_of_faces()

    def number_of_cells(self) -> int:
        return self.ds.number_of_cells()

    def uniform_refine(self) -> None:
        raise NotImplementedError

    def cell_location(self):
        raise NotImplementedError


    ### FEM Interfaces ###

    def geo_dimension(self) -> int:
        """
        @brief Get geometry dimension of the mesh.
        """
        return self.node.shape[-1]

    def top_dimension(self) -> int:
        """
        @brief Get topology dimension of the mesh.
        """
        return self.ds.TD

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype: Union[int, str], index=np.s_[:]):
        raise NotImplementedError

    def entity_barycenter(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_measure(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def integrator(self, k):
        raise NotImplementedError

    def shape_function(self, p):
        raise NotImplementedError

    def grad_shape_function(self, p, index=np.s_[:]):
        raise NotImplementedError

    def number_of_local_ipoints(self, p):
        raise NotImplementedError

    def number_of_global_ipoints(self, p):
        raise NotImplementedError

    def interpolation_points(self):
        raise NotImplementedError

    def cell_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def edge_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def face_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def node_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError


# TODO: Is this better?

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


class _Entity():
    pass

# Similarly to the example above,
# Implement `_Measure`, `_Barycenter` class for every types of meshes.

class _Measure():
    pass

class _Barycenter():
    pass

# Mesh typs has `measure` and `barycenter` properties (point to the classes' instances above)
# to specify how to measure entities, and how to calculate the barycenter of entities.


class _Boundary():
    def __init__(self) -> None:
        pass

    def edge_flag(self):
        pass


class _Plot():
    pass


# Structure:
# mesh.torch Module
#  |- Mesh, Mesh1d, Mesh2d, Mesh3d -> mesh.py
#  |- MeshDataStructure, ... -> mesh_data_structure.py
#  |- Entity, Measure, Barcenter, ... -> typing.py
#  |- TriangleMesh, TriangleMeshDataStructure, _Entity, ... -> triangle_mesh.py
