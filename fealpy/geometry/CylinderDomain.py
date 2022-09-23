
import numpy as np

from .sizing_function import huniform

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
