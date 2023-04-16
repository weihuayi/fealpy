
from typing import Optional, Generic, TypeVar
from torch import Tensor
import numpy as np


# Descriptor for entities

_VT = TypeVar('_VT')

class Redirector(Generic[_VT]):
    def __init__(self, target: str) -> None:
        self._target = target

    def __get__(self, obj, objtype) -> _VT:
        return getattr(obj, self._target)

    def __set__(self, obj, val: _VT):
        setattr(obj, self._target, val)


class MeshDataStructure():
    NN: int = -1
    TD: int

    cell: Tensor
    face: Optional[Tensor]
    edge: Optional[Tensor]
    edge2cell: Optional[Tensor]

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


class Mesh():
    ds: MeshDataStructure
    node: Tensor

    ### Defined here ###

    def top_dimension(self):
        return self.ds.TD

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.number_of_edges()

    def number_of_faces(self) -> int:
        return self.ds.number_of_faces()

    def number_of_cells(self) -> int:
        return self.ds.number_of_cells()

    ### Defined in Meshxd ###

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_barycenter(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_measure(self, etype, index=np.s_[:]):
        raise NotImplementedError

    ### Defined in final Mesh.

    def geo_dimension(self) -> int:
        raise NotImplementedError

    def integrator(self, k):
        raise NotImplementedError

    def uniform_refine(self):
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
