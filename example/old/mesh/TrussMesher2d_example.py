import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TriMesher import trussmesh2d


class RectangleDomain:
    def __init__(self, box):
        """
        """
        self.box = box
        self.bbox = [box[0]-0.1, box[1]+0.1, box[2]-0.1, box[3]+0.1] 

        vertices = np.array([
            (box[0], box[2]),
            (box[1], box[2]),
            (box[1], box[3]),
            (box[0], box[3]),
            ], dtype=np.float64)

        curves = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            ], dtype=np.float64)

        self.facets = {0:vertices, 1:curves}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        x = p[..., 0]
        y = p[..., 1]
        d = dmin(y - box[2], box[3] - y)
        d = dmin(d, x - box[0])
        d = dmin(d, box[1] - x)
        return -d

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        pass


