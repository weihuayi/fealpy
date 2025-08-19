
from typing import Any

from ..backend import bm
from ..decorator import variantmethod

from gmsh import model

class DLDModeler:
    """
    微流芯片网格几何建模类
    
    参数如下:
        bar: 初始缓存区对应 Box
        m: 单个周期的 Box 每行立柱个数
        n: 单个周期的 Box 每列立柱个数
        theta: 偏转角
        radius: 立柱半径
        T: 周期数
    
    
    """
    def __init__(self, options):
        self.options = options

    def get_options(self) -> dict:
        options = {
            'bar': [0.0, 1, 0.0, 5],
            'm': 6,
            'n': 8,
            'theta': bm.pi / 10,
            'radius': 0.1,
            'T': 3
        }

        return options
      
    @variantmethod('circle')
    def build(self, gmsh=None):
        option = self.options
        bar = option['bar']
        m = option['m']
        n = option['n']
        theta = option['theta']
        r = option['radius']
        T = option['T']

        x0, x1, y0, y1 = bar
        cy0 = (y1 - y0) / (n + 1) + y0
        l2 = (y1 - y0) / (n + 1)
        h = l2 / m
        l1 = h / bm.tan(theta)
        cx0 = l1 / 3 + x1

        gmsh.model.add("DLD")
        rectangle = gmsh.model.occ.addRectangle(
            x0, y0, 0, 
            2 * (x1 - x0) + (T - 1) * m * l1 + (m - 1 / 3) * l1, 
            y1 - y0 + h / l1 * ((m - 1 / 3) * l1)
        )

        centers = []
        col_idx = 0
        x = cx0
        
        for i in range(m):
            y_start = cy0 + col_idx * h
            for j in range(n):
                y = y_start + j * l2
                for k in range(T):
                    centers.append((x + k * m * l1, y))

            x += l1
            col_idx += 1
            
        self.centers = centers

        circle_tags = []
        for cx, cy in centers:
                circle = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
                circle_tags.append(circle)

        gmsh.model.occ.cut([(2, rectangle)], [(2, tag) for tag in circle_tags], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()   

    @build.register('ellipse')
    def build(self, gmsh=None):
        pass

    @build.register('droplet')
    def build(self, gmsh=None):
        pass

    @build.register('triangle')
    def build(self, gmsh=None):
        pass

