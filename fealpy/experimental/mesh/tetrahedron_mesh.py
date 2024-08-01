from ..backend import backend_manager as bm
from .mesh_base import SimplexMesh

class TetrahedronMesh(SimplexMesh): 
    def __init__(self, node, cell) 
        super(TetrahedronMesh, self).__init__(TD=3)
        self.node = node
        self.cell = cell
        self.localFace = bm.array([(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)])
        self.localEdge = bm.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self.localCell = bm.array([
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)])




