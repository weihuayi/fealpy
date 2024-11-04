import numpy as np

from fealpy.mesh import HalfFacePolyhedronMesh

class HalfFacePolygonMeshDataStructure():

    def __init__(self):
        pass

    def from_mesh_test(self):
        node = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]], dtype=np.float)

        cell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int)
        tmesh = TetrahedronMesh(node, cell)
        pmesh = HalfFacePolyhedronMesh.from_mesh(tmesh)

        


test = HalfFacePolyhedronMesh()
test.from_mesh_test()
