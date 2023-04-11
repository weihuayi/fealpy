
from typing import Optional
from torch import Tensor
import numpy as np


# Descriptor for entities
class Entity():
    def __init__(self) -> None:
        self._data = None

    def __get__(self, obj, objtype) -> Tensor:
        if self._data is not None:
            return self._data
        raise ValueError(f"No data for the entity.")

    def __set__(self, obj, val: Tensor):
        self._data = val


class MeshDataStructure():
    NN: int = -1
    '''Number of nodes'''
    # NE: int = -1
    # '''Number of edges'''
    # NF: int = -1
    # '''Number of faces'''
    # NC: int = -1
    # '''Number of cells'''
    cell = Entity()
    face = Entity()
    edge = Entity()
    edge2cell = Entity()

    localEdge: Tensor
    localFace: Tensor
    localCell: Tensor

    NVC: int
    NVE: int
    NVF: int
    NEC: int
    NFC: int

    def __init__(self, NN: int, cell: Tensor):
        self.reinit(NN=NN, cell=cell)
        self.itype = cell.dtype
        self.device = cell.device

    def reinit(self, NN: int, cell: Tensor):
        self.NN = NN
        self.cell = cell
        self.construct()

    @property
    def NC(self):
        """Number of cells"""
        return self.cell.shape[0]

    @property
    def NF(self):
        """Number of faces"""
        return self.face.shape[0]

    @property
    def NE(self):
        """Number of edges"""
        return self.edge.shape[0]


class Mesh():
    ds: MeshDataStructure
    node: Tensor

    ### Defined here ###

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.NE

    def number_of_faces(self) -> int:
        return self.ds.NF

    def number_of_cells(self) -> int:
        return self.ds.NC

    ### Defined in Meshxd ###

    def top_dimension(self) -> int:
        raise NotImplementedError

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
