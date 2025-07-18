from typing import Sequence
from ...backend import backend_manager as bm
from ...decorator import variantmethod
from ...mesh import TetrahedronMesh

class SphereShellMesher:
    """Sphere shell mesh generator"""
    def __init__(self, h=None):

        if h is None:
            self.h = 0.3
        else:
            self.h = h

    def geo_dimension(self) -> int:
        return 3

    @variantmethod('shell')
    def init_mesh(self, r1=0.05, r2=0.5): 
        mesh = TetrahedronMesh.from_spherical_shell(r1=r1, r2=r2, h=self.h)
        return mesh

    @init_mesh.register('tet')
    def init_mesh(self): 
        mesh = TetrahedronMesh.from_unit_sphere_gmsh(h=self.h)
        return mesh