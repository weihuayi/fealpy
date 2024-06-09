import numpy as np

from fealpy.geometry.domain import Domain
from fealpy.geometry import huniform
from fealpy.geometry import drectangle,dcircle,ddiff

class Rectangle_BJT_Domain(Domain):
    def __init__(self, domain=[0, 100, 0, 220], fh=None):
        """
        """
        super().__init__(GD=2)
        if fh is not None:
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
        return drectangle(p, self.domain)

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

class RB_IGCT_Domain(Domain):
    def __init__(self, fh=None):
        """
        """
        super().__init__(GD=2)
        if fh is not None:
            self.fh = fh

        self.box = [-100, 345, -100, 1600]

        vertices = np.array([
            (0.0,0.0),
            (92.0,0.0),
            (110.0, 18.0),
            (245.0,18.0),
            (245.0,1500.0),
            (0.0,1500.0)
            ], dtype=np.float64)

        circenter = np.array([(110.0,0.0)],dtype=np.float64)
        
        lines = np.array([
            (0, 1),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0)], dtype=np.float64)

        circleArc = np.array([(1,2)],dtype=np.float64)

        self.facets = {0:vertices, 1:lines, 2:circleArc, 10:circenter}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        circenter = np.array([110.0,0.0],dtype=np.float64)
        d0 = drectangle(p,[0.0,245.0,0.0,1500.0])
        d1 = drectangle(p,[110.0,245.0,0.0,18.0])
        d2 = dcircle(p,cxy=circenter,r=18)
        d3 = ddiff(d0,d1)
        return ddiff(d3,d2)

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

