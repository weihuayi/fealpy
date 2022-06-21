import numpy as np

from .sizing_function import huniform
from .implicit_surface import SphereSurface

class SphereDomain():
    def __init__(self, c=np.array([0.0, 0.0, 0.0]), r=1.0, fh=huniform):

        self.surface = SphereSurface(c, r)
        self.box = self.surface.box
        self.fh = fh

        self.facets = {0:None, 1:None, 2:self.surface}

    def __call__(self, p):
        """
        @brief 描述区域符号距离函数
        """
        return self.surface(p)

    def signed_dist_function(self, p):
        return self(p)

    def sizing_function(self, p):
        return self.fh(p)

    def facet(self, dim):
        return self.facets[dim]

    def projection(self, p):
        p, d = self.surface.project(p)
        return p


