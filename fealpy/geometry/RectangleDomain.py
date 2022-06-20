import numpy as np

from .sizing_function import huniform
from .signed_distance_function import dmin

class RectangleDomain:
    def __init__(self, domain=[0, 1, 0, 1], fh=huniform):
        """
        """
        self.fh = fh
        self.domain = domain 

        mx = (domain[1] - domain[0])/10
        my = (domain[3] - domain[2])/10
        self.box = [domain[0]-mx, domain[1]+mx, domain[2]-my, domain[3]+my] 

        vertices = np.array([
            (domain[0], domain[2]),
            (domain[1], domain[2]),
            (domain[1], domain[3]),
            (domain[0], domain[3]),
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
        domain = self.domain
        x = p[..., 0]
        y = p[..., 1]
        d = dmin(y - domain[2], domain[3] - y)
        d = dmin(d, x - domain[0])
        d = dmin(d, domain[1] - x)
        return -d

    def signed_dist_function(self, p):
        return self(p)

    def sizing_function(self, p):
        return self.fh(p)

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        pass
