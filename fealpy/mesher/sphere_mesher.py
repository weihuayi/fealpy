from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..geometry import SphereSurface
from ..mesh import TetrahedronMesh,QuadrangleMesh,TriangleMesh
from ..mesh import LagrangeQuadrangleMesh,LagrangeTriangleMesh

class SphereVolumnMesher:
    """Sphere domain mesh generator"""
    def __init__(self):
        self.surface = SphereSurface()

    def geo_dimension(self) -> int:
        return 3

    @variantmethod('tet')
    def init_mesh(self, h=0.1): 
        mesh = TetrahedronMesh.from_unit_sphere_gmsh(h=self.h)
        return mesh
    

class SphereSurfaceMesher:
    """Sphere domain mesh generator"""
    def __init__(self):
        self.surface = SphereSurface()
        
    def geo_dimension(self) -> int:
        return 3

    @variantmethod('ltri')
    def init_mesh(self, p:int=1):
        """Create a LagrangeTriangleMesh from a triangle mesh."""
        lmesh = TriangleMesh.from_unit_sphere_surface()
        mesh = LagrangeTriangleMesh.from_triangle_mesh(lmesh, p=p, surface=self.surface)
        return mesh
    
    @variantmethod('lquad')
    def init_mesh(self, p:int=1):
        """Create a LagrangeQuadrangleMesh from a quadrangle mesh."""
        lmesh = QuadrangleMesh.from_unit_sphere_surface()
        mesh = LagrangeQuadrangleMesh.from_quadrangle_mesh(lmesh, p=p, surface=self.surface)
        return mesh
