
from torch import Tensor, dtype

class MeshDataStructure():
    NN: int = -1
    '''Number of nodes'''
    NE: int = -1
    '''Number of edges'''
    NF: int = -1
    '''Number of faces'''
    NC: int = -1
    '''Number of cells'''
    cell: Tensor
    edge: Tensor
    itype: dtype

    localEdge: Tensor
    localFace: Tensor
    localCell: Tensor

    NVC: int
    NVE: int
    NVF: int
    NEC: int
    NFC: int


class Mesh():
    ds: MeshDataStructure
    node: Tensor

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.NE

    def number_of_faces(self) -> int:
        return self.ds.NF

    def number_of_cells(self) -> int:
        return self.ds.NC

    def geo_dimension(self) -> int:
        raise NotImplementedError

    def top_dimension(self) -> int:
        raise NotImplementedError

    def uniform_refine(self):
        raise NotImplementedError

    def integrator(self, k):
        raise NotImplementedError

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_barycenter(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_measure(self, etype, index=np.s_[:]):
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
