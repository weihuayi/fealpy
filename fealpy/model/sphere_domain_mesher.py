from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TetrahedronMesh

class SphereDomainMesher:
    """Sphere domain mesh generator"""
    def __init__(self, h=None):

        if h is None:
            self.h = 0.3
        else:
            self.h = h

    def geo_dimension(self) -> int:
        return 3

    @variantmethod('tet')
    def init_mesh(self): 
        mesh = TetrahedronMesh.from_unit_sphere_gmsh(h=self.h)
        return mesh
