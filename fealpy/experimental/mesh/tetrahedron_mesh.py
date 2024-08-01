from ..backend import backend_manager as bm
from .mesh_base import SimplexMesh

class TetrahedronMesh(SimplexMesh): 

    def __init__(self, node, cell):
        super(TetrahedronMesh, self).__init__(TD=3)
        self.node = node
        self.cell = cell

        self.meshtype = 'tet'
        self.p = 1 # linear mesh

        kwargs = {"dtype": self.cell.dtype, } # TODO: 增加 device 参数
        self.localEdge = bm.tensor([
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], **kwargs)
        self.localFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **kwargs)
        self.localCell = bm.tensor([
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)], **kwargs)

        self.ccw = bm.tensor([0, 1, 2, 4], **kwargs)
        self.construct()

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {} 
        self.celldata = {}
        self.meshdata = {}

    ## @ingroup MeshGenerators
    @classmethod
    def from_one_tetrahedron(cls, meshtype='equ'):
        """
        """
        if meshtype == 'equ':
            node = bm.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, bm.sqrt(3)/2, 0.0],
                [0.5, bm.sqrt(3)/6, bm.sqrt(2/3)]], dtype=bm.float64)
        elif meshtype == 'iso':
            node = bm.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=bm.float64)
        cell = bm.array([[0, 1, 2, 3]], dtype=bm.int32)
        return cls(node, cell)


