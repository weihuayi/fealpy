import taichi as ti

class TetrahedronMeshDataStructure():

    localFace = ti.Matrix([(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)])
    localEdge = ti.Matrix([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    localFace2edge = ti.Matrix([(5, 4, 3), (5, 1, 2), (4, 2, 0), (3, 0, 1)])
    index = ti.Matrix([
       (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
       (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
       (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
       (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)]);
    NVC = 4
    NEC = 6
    NFC = 4
    NVF = 3
    NEF = 3

    def __init__(self, NN, cell):
        NC = cell.shape[0]
        self.cell = ti.field(ti.i32, shape=(NC, 4))

@ti.data_oriented
class TetrahedronMesh():
    def __init__(self, node, cell):
        assert cell.shape[-1] == 3
        NN = node.shape[0]
        GD = node.shape[1]
        self.node = ti.field(ti.float64, (NN, GD))
        self.node.from_numpy(node)
        self.ds = TriangleMeshDataStructure(NN, cell);
