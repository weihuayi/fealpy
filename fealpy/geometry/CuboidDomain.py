
import numpy as np

from .signed_distance_function import dmin
from .sizing_function import huniform

class CuboidDomain():
    def __init__(self, domain=[0, 1, 0, 1, 0, 1], fh=huniform):

        self.domain = domain 

        hx = domain[1] - domain[0]
        hy = domain[3] - domain[2]
        hz = domain[5] - domain[4]

        m = np.min([hx, hy, hz])/11

        self.box = [
                domain[0]-m, domain[1]+m,
                domain[2]-m, domain[3]+m,
                domain[4]-m, domain[5]+m
                ]

        self.fh = fh

        facet0 = np.array([
            (domain[0], domain[2], domain[4]),
            (domain[1], domain[2], domain[4]),
            (domain[1], domain[3], domain[4]),
            (domain[0], domain[3], domain[4]),
            (domain[0], domain[2], domain[5]),
            (domain[1], domain[2], domain[5]),
            (domain[1], domain[3], domain[5]),
            (domain[0], domain[3], domain[5])
            ], dtype=np.float64)

        facet1 = np.array([
            (0, 1), (1, 2), (2, 3), (0, 3),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (4, 7)])

        facet2 = np.array([
            (0, 3, 2, 1), (4, 5, 6, 7), # bottom and top faces
            (0, 4, 7, 3), (1, 2, 6, 5), # left and right faces  
            (0, 1, 5, 4), (2, 3, 7, 6)])# front and back faces

        self.facets = {0:facet0, 1:facet1, 2:facet2}

    def __call__(self, p):
        """
        @brief 描述区域符号距离函数
        """
        domain = self.domain
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        d = -dmin(
                z - domain[4], domain[5] - z, 
                y - domain[2], domain[3] - y, 
                x - domain[0], domain[1] - x)

        # (0, 1)
        val0 = domain[2] - y
        val1 = domain[4] - z
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (1, 2)
        val0 = x - domain[1] 
        val1 = domain[4] - z
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (2, 3)
        val0 = y - domain[3] 
        val1 = domain[4] - z
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (0, 3)
        val0 = domain[0] - x
        val1 = domain[4] - z
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (0, 4)
        val0 = domain[0] - x
        val1 = domain[2] - y 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (1, 5)
        val0 = x - domain[1]
        val1 = domain[2] - y 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (2, 6)
        val0 = x - domain[1]
        val1 = y - domain[3] 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (3, 7)
        val0 = domain[0] - x
        val1 = y - domain[3] 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (4, 5)
        val0 = domain[2] - y 
        val1 = z - domain[5] 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (5, 6)
        val0 = x - domain[1] 
        val1 = z - domain[5] 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (6, 7)
        val0 = y - domain[3] 
        val1 = z - domain[5] 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        # (4, 7)
        val0 = domain[0] - x 
        val1 = z - domain[5] 
        flag = (val0 > 0) & (val1 > 0)
        d[flag] = np.sqrt(val0[flag]**2 + val1[flag]**2)

        return d

    def signed_dist_function(self, p):
        return self(p)

    def sizing_function(self, p):
        return self.fh(p, self)

    def facet(self, dim):
        return self.facets[dim]
