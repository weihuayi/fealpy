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


class SphereDomain():
    def __init__(self, c=np.array([0.0, 0.0, 0.0]), r=1.0, fh=huniform):
        from .implicit_surface import SphereSurface

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

class CylinderDomain():
    def __init__(self, 
            c=np.array([0.0, 0.0, 0.0]), 
            h=2.0, 
            r=1.0,
            d=np.array([0.0, 0.0, 1.0]), fh=huniform, pfix=None):

        self.center = c # 圆柱中心的位置
        self.height = h # 圆柱的长度
        self.radius = r # 圆柱的半径

        l = np.sqrt(np.sum(d**2))
        self.direction = d/l # 圆柱的方向, 单位向量 

        self.box = [c[0]-h-r, c[0]+h+r, c[1]-h-r, c[1]+h+r, c[2]-h-r, c[2]+h+r]

        self.fh = fh
        self.facets = {0:pfix, 1:None}

    def __call__(self, p):
        """
        @brief 描述区域符号距离函数
        """
        v = p - self.center
        d = np.sum(v*self.direction, axis=-1) # 中轴方向上到中心点的距离
        v -= d[..., None]*self.direction # v: 到中轴的距离

        shape = p.shape[:-1] + (3, )
        val = np.zeros(shape, dtype=p.dtype)
        val[..., 0] = np.sqrt(np.sum(v**2, axis=1)) - self.radius # 到圆柱面的距离
        val[..., 1] =  d - self.height/2 # 到圆柱上圆面的距离
        val[..., 2] = -d - self.height/2 # 到圆柱下圆面的距离

        d = np.max(val, axis=-1)

        flag = (val[..., 0] > 0) & (val[..., 1] > 0)
        d[flag] = np.sqrt(val[flag, 0]**2 + val[flag, 1]**2)
        flag = (val[..., 0] > 0) & (val[..., 2] > 0)
        d[flag] = np.sqrt(val[flag, 0]**2 + val[flag, 2]**2)
    
        return d

    def signed_dist_function(self, p):
        return self(p)

    def sizing_function(self, p):
        return self.fh(p, self)

    def facet(self, dim):
        return self.facets[dim]


class TorusDomain():
    def __init__(self,fh=huniform):
        from .implicit_surface import TorusSurface
        self.surface = TorusSurface()
        self.box = self.surface.box
        self.fh = fh

        self.facets = {0:None,1:None}
    def __call__(self,p):
        return self.surface(p)

    def signed_dist_function(self,p):
        return self(p)
    
    def sizing_function(self,p):
        return self.fh(p)
    
    def facet(self,dim):
        return self.facets[dim]
    
    def projection(self,p):
        p,d = self.surface.project(p)
        return p
