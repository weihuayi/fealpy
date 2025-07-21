from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import HexahedronMesh, TetrahedronMesh

class CylinderMesher:
    """Cylinder domain mesh generator"""
    def __init__(self, radius=None, height=None, lc=None):
        
        if radius is None:
            self.radius = 1
        else:
            self.radius = radius
        
        if height is None:
            self.height = 1
        else:
            self.height = height
        
        if lc is None:
            self.lc = 0.3
        else:
            self.lc = lc
        
    def geo_dimension(self) -> int:
        return 3

    @variantmethod('tet')
    def init_mesh(self): 
        mesh = TetrahedronMesh.from_cylinder_gmsh(
            radius=self.radius, 
            height=self.height,
            lc=self.lc)
        return mesh

