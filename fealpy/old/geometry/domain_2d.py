import numpy as np

from .sizing_function import huniform
from .signed_distance_function import dmin, dcircle, drectangle, ddiff
from .geoalg import project
from .implicit_curve import CircleCurve
from .domain import Domain

class RectangleDomain(Domain):
    def __init__(self, domain=[0, 1, 0, 1], hmin=0.1, hmax=None, fh=None):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
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

class CircleDomain(Domain):
    def __init__(self, center=[0.0, 0.0], radius=1.0, hmin=0.1, hmax=0.1,
            fh=None):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
        if fh is not None:
            self.fh = fh

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

    def project(self, p):
        p, d = self.curve.project(p)
        return p

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        pass

class LShapeDomain(Domain):
    def __init__(self, hmin=0.1, hmax=0.1, fh=None):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
        if fh is not None:
            self.fh = fh

        self.box = [-1.0, 1.0, -1.0, 1.0] 
        self.fh = fh
        vertices = np.array([
            (-1.0, -1.0),
            ( 0.0, -1.0),
            ( 0.0,  0.0),
            ( 1.0,  0.0),
            ( 1.0,  1.0),
            (-1.0,  1.0)
            ], dtype=np.float64)

        curves = np.array([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0)
            ], dtype=np.int_)

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

    def sizing_function(self, p):
        return self.fh(p, self)

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        pass

class SquareWithCircleHoleDomain(Domain):
    def __init__(self, hmin=0.1, hmax=0.1, fh=None):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
        if fh is not None:
            self.fh = fh

        self.box = [0, 1, 0, 1]
        vertices = np.array([
            (0.0, 0.0), 
            (1.0, 0.0), 
            (1.0, 1.0),
            (0.0, 1.0)],dtype=np.float64)

        fd1 = lambda p: dcircle(p, [0.5, 0.5], 0.2)
        fd2 = lambda p: drectangle(p, [0.0, 1, 0.0, 1])
        fd = lambda p: ddiff(fd2(p), fd1(p))

        self.facets = {0:vertices, 1:fd}


    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        return self.facets[1](p) 

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

class BoxWithCircleHolesDomain(Domain):
    def __init__(self, box=[0, 1, 0, 1], circles=[(0.5, 0.5, 0.2)], hmin=0.003,
            hmax=0.01):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)

        self.box = box 
        vertices = np.array([
            (box[0], box[2]), 
            (box[1], box[2]), 
            (box[1], box[3]),
            (box[0], box[3])],dtype=np.float64)

        self.circles = []
        for val in circles:
            self.circles.append(lambda p: dcircle(p, val[0:2], val[2]))

        def fd(p):
            d0 = drectangle(p, box)
            for circle in self.circles:
                d0 = ddiff(d0, circle(p))
            return d0

        def fh(p):
            d0 = 1e10 
            for circle in self.circles:
                d0 = np.minimum(d0, circle(p))
            h = hmin + 0.05*d0
            h[h>hmax] = hmax 
            return h

        self.fh = fh
        self.facets = {0:vertices, 1:fd}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        return self.facets[1](p) 

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

class BoxWithBoxHolesDomain(Domain):
    def __init__(self, box=[-2, 2, -2, 2], boxs=[(-1, 1, -1, 1)], hmin=0.1,
            hmax=0.1, fh=None):
        """
        """
        super().__init__(hmin=hmin, hmax=hmax, GD=2)

        self.box = box 
        self.boxs = boxs

        nv = 4*(len(boxs)+1)
        vertices = np.zeros((nv, 2), dtype=np.float64)
        vertices[0:4] = np.array([
            (box[0], box[2]), 
            (box[1], box[2]), 
            (box[1], box[3]),
            (box[0], box[3])], dtype=np.float64)

        start = 4
        end = 8
        for b in boxs:
            vertices[start:end] = np.array([
                (b[0], b[2]), 
                (b[1], b[2]), 
                (b[1], b[3]),
                (b[0], b[3])], dtype=np.float64)
            start = end
            end += 4 

        def fd(p):
            d0 = drectangle(p, box)
            for b in boxs:
                # d0 = ddiff(d0, drectangle(p, b)) #TODO: 为什么这样写是错的呢？
                d1 = drectangle(p, b)
                d0 = ddiff(d0, d1)
            return d0

        def fh(p):
            d0 = 1e10 
            for b in boxs:
                d0 = np.minimum(d0, drectangle(p, b))
            h = hmin + 0.05*d0
            h[h>hmax] = hmax 
            return h

        self.fh = fh

        self.facets = {0:vertices, 1:fd}

    def __call__(self, p):
        """
        @brief 符号距离函数
        """
        return self.facets[1](p) 

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
