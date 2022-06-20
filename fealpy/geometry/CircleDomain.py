import numpy as np

from .sizing_function import huniform
from .signed_distance_function import dmin

class CircleDomain:
    def __init__(self, center=[0.0, 0.0], r=1.0, fh=huniform):
        """
        """
        self.center = center
        self.r = r
        self.fh = fh

        margin = r/10
        self.bbox = [c[0]-r-margin, c[0]+r+margin, c[1]-r-margin, c[1]+r+margin] 

        self.facets = {0:None, 1:None}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        box = self.box
        x = p[..., 0]
        y = p[..., 1]
        d = dmin(y - box[2], box[3] - y)
        d = dmin(d, x - box[0])
        d = dmin(d, box[1] - x)
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
