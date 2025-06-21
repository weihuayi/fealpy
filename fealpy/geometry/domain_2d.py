
from ..backend import backend_manager as bm
from .sizing_function import huniform
from .signed_distance_function import dmin, dcircle, drectangle, ddiff
from .geoalg import project
from .implicit_curve import CircleCurve
from .domain import Domain

class LShapeDomain(Domain):
    def __init__(self, hmin=0.1, hmax=0.1, fh=None):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
        if fh is not None:
            self.fh = fh

        self.box = [-1.0, 1.0, -1.0, 1.0] 
    
        vertices = bm.array([
            (-1.0, -1.0),
            ( 0.0, -1.0),
            ( 0.0,  0.0),
            ( 1.0,  0.0),
            ( 1.0,  1.0),
            (-1.0,  1.0)
            ], dtype=bm.float64)

        curves = bm.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0)
            ], dtype=bm.int32)

        self.facets = {0:vertices, 1:curves}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        d0 = drectangle(p, [-1.0, 1.0, -1.0, 1.0])
        d1 = drectangle(p, [ 0.0, 1.0, -1.0, 0.0])
        return ddiff(d0, d1)

    def signed_dist_function(self, p):
        return self(p) 

    def grad_signed_dist_function(self, p, deps=1e-12):
        """
        """
        d = self(p)
        kargs = bm.context(p)
        depsx = bm.array([deps, 0], **kargs)
        depsy = bm.array([0, deps], **kargs)
        
        dgradx = (self(p + depsx) - d)/deps
        dgrady = (self(p + depsy) - d)/deps

        n = bm.stack([dgradx, dgrady], axis=1)
        return n


    def sizing_function(self, p):
        return self.fh(p, self)

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        pass
