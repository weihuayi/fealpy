import numpy as np

from .sizing_function import huniform
from .signed_distance_function import dmin

class BoxDomain():
    def __init__(self, box, fh=huniform):
        self.box = box
        self.bbox = [
                box[0]-0.1, box[1]+0.1, 
                box[2]-0.1, box[3]+0.1,
                box[4]-0.1, box[5]+0.1] 

        self.fh = fh

        vertices = np.array([
            (box[0], box[2], box[4]),
            (box[1], box[2], box[4]),
            (box[1], box[3], box[4]),
            (box[0], box[3], box[4]),
            (box[0], box[2], box[5]),
            (box[1], box[2], box[5]),
            (bxo[1], box[3], box[5]),
            (box[0], box[3], box[5])], dtype=np.float64)

        curves = np.array([
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (7, 4)], dtype=np.int_)

        surfaces = np.array([
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0)], dtype=np.int_)

        self.facets = {0:vertices, 1:curves, 2:surfaces}

    def __call__(self, p):
        """
        @brief 描述区域符号距离函数
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        d = dmin(z - box[4], box[5] - z)
        d = dmin(d, y - box[2])
        d = dmin(d, box[3] - y)
        d = dmin(d, x - box[0])
        d = dmin(d, box[1] - x)
        return -d

    def signed_dist_function(self, p):
        return self(p)

    def sizing_function(self, p):
        return self.fh(p)

    def facet(self, dim):
        return self.facets[dim]

