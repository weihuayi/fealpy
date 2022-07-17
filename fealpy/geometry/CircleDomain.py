import numpy as np

from .sizing_function import huniform
from .signed_distance_function import dmin, dcircle
from .implicit_curve import CircleCurve

class CircleDomain:
    def __init__(self, center=[0.0, 0.0], radius=1.0, fh=huniform):
        """
        """
        self.curve = CircleCurve(center, radius) 
        self.fh = fh 
        m = radius + radius/10
        self.box = [center[0]-m, center[0]+m, center[1]-m, center[1]+m] 

        self.facets = {0:None, 1:None}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        return self.curve(p)

    def signed_dist_function(self, p):
        return self.curve(p) 

    def sizing_function(self, p):
        return self.fh(p, self)

    def projection(self, p):
        p, d = self.curve.project(p)
        return p

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        pass
