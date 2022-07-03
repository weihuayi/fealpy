
import numpy as np

from .sizing_function import huniform

class SphereDomain():
    def __init__(self, 
            c=np.array([0.0, 0.0, 0.0]), 
            h=1.0, 
            r=1.0,
            d=np.array([0.0, 0.0, 1,0]), fh=huniform):

        self.center = c
        self.height = h
        self.radius = r
        self.direction = d

        self.box = 

        self.fh = fh
        self.facets = {0:None, 1:None}

    def __call__(self, p):
        """
        @brief 描述区域符号距离函数
        """
        v = p - self.center
        d = np.sum(v*self.direction, axis=-1)
        v -= d[..., None]*self.direction

        shape = p.shape[:-1] + (5, )
        val = np.zeros(shape, dytpe=p.dtype)
        val[..., 0] = np.sqrt(np.sum(v**2, axis=1)) - self.radius
        val[..., 1] =  d - self.height
        val[..., 2] = -d - self.height 
        val[..., 3] = np.sqrt(val[..., 0]**2 + val[..., 1]**2)
        val[..., 4] = np.sqrt(val[..., 0]**2 + val[..., 2]**2)

        d = np.max(val[..., 0:3], axis=-1)

        flag = (val[..., 0] > 0) & (val[..., 1] > 0)
        d[flag] = val[flag, 3]
        flag = (val[..., 0] > 0) & (val[..., 2] > 0)
        d[flag] = val[flag, 4]
    
        return d

    def signed_dist_function(self, p):
        return self(p)

    def sizing_function(self, p):
        return self.fh(p)

    def facet(self, dim):
        return self.facets[dim]
