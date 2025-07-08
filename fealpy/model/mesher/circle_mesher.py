from typing import Sequence
from ...backend import backend_manager as bm
from ...decorator import variantmethod
from ...mesh import TriangleMesh, QuadrangleMesh

class CircleMesher:
    """Circle domain mesh generator"""
    def __init__(self, vertices=None, h=None):
        if h is None:
            self.h = 0.1
        else:
            self.h = h

        if vertices == None:
            n = int(2 * bm.pi / self.h)
            delta_theta = 2 * bm.pi / n 
            theta = bm.linspace(0, 2 * bm.pi - delta_theta, n, endpoint=True)
            x = bm.cos(theta)
            y = bm.sin(theta)
            self.vertices = bm.stack([x, y]).T
        else:
            self.vertices = vertices

    def geo_dimension(self) -> int:
        return 2

    @variantmethod('unit_tri')
    def init_mesh(self): 
        mesh = TriangleMesh.from_unit_circle_gmsh(h=self.h)
        return mesh 

    @init_mesh.register('tri')
    def init_mesh(self): 
        mesh = TriangleMesh.from_polygon_gmsh(vertices=self.vertices, h=self.h)
        return mesh   

    @init_mesh.register('quad')
    def init_mesh(self): 
        mesh = QuadrangleMesh.from_polygon_gmsh(vertices=self.vertices, h=self.h)
        return mesh  
