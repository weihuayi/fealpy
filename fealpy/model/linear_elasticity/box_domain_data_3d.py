
from ...decorator import cartesian, variantmethod
from ...backend import bm
from ...typing import TensorLike
from ...material import LinearElasticMaterial


class BoxDomainData3d():

    def __init__(self):
        self.eps = 1e-12


    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    @variantmethod('hex')
    def init_mesh(self):
        from ...mesh import HexahedronMesh
        node = bm.array([[0.249, 0.342, 0.192],
                [0.826, 0.288, 0.288],
                [0.850, 0.649, 0.263],
                [0.273, 0.750, 0.230],
                [0.320, 0.186, 0.643],
                [0.677, 0.305, 0.683],
                [0.788, 0.693, 0.644],
                [0.165, 0.745, 0.702],
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]],
            dtype=bm.float64)

        cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                        [0, 3, 2, 1, 8, 11, 10, 9],
                        [4, 5, 6, 7, 12, 13, 14, 15],
                        [3, 7, 6, 2, 11, 15, 14, 10],
                        [0, 1, 5, 4, 8, 9, 13, 12],
                        [1, 2, 6, 5, 9, 10, 14, 13],
                        [0, 4, 7, 3, 8, 12, 15, 11]],
                        dtype=bm.int32)
        mesh = HexahedronMesh(node, cell)
        return mesh
