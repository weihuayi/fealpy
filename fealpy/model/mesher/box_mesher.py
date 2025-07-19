from typing import Sequence
from ...backend import backend_manager as bm
from ...decorator import variantmethod
from ...mesh import HexahedronMesh, TetrahedronMesh, QuadrangleMesh, TriangleMesh

class BoxMesher2d:
    """Box domain mesh generator"""
    def __init__(self, box=None):
        if box is None:
            self.box = [0, 1, 0, 1]
        else:
            self.box = box

    def geo_dimension(self) -> int:
        return 2
    
    def domain(self) -> Sequence[float]:
        return self.box

    @variantmethod('uniform_quad')
    def init_mesh(self, nx=10, ny=10):
        mesh = QuadrangleMesh.from_box(box=self.box, nx=nx, ny=ny)
        return mesh
    
    @init_mesh.register('uniform_tri')
    def init_mesh(self, nx=10, ny=10):
        mesh = TriangleMesh.from_box(box=self.box, nx=nx, ny=ny)
        return mesh

    @init_mesh.register('moving_tri')
    def init_mesh(self, nx=64, ny=64,**kwargs):
        mesh = TriangleMesh.from_box(self.box, nx=nx, ny=ny, **kwargs)
        domain = self.box
        vertices = bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], **kwargs)
        mesh.nodedata['vertices'] = vertices
        return mesh
    @init_mesh.register('moving_quad')
    def init_mesh(self ,nx = 64, ny = 64, **kwargs):
        mesh = QuadrangleMesh.from_box(self.box, nx=nx, ny=ny, **kwargs)
        domain = self.box
        vertices = bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], **kwargs)
        mesh.nodedata['vertices'] = vertices
        return mesh

class BoxMesher3d:
    """Box domain mesh generator"""
    def __init__(self, box=None):

        if box is None:
            self.box = [0, 1, 0, 1, 0, 1]
        else:
            self.box = box

    def geo_dimension(self) -> int:
        return 3
    
    def domain(self) -> Sequence[float]:
        return self.box

    @variantmethod('uniform_tet')
    def init_mesh(self, nx=10, ny=10, nz=10): 
        mesh = TetrahedronMesh.from_box(box=self.box, nx=nx, ny=ny, nz=nz)
        return mesh
    
    @init_mesh.register('uniform_hex')
    def init_mesh(self, nx=10, ny=10, nz=10):
        mesh = HexahedronMesh.from_box(box=self.box, nx=nx, ny=ny, nz=nz)
        return mesh
    
    @init_mesh.register('seven_hex')
    def init_mesh(self):
        node = bm.array([[0.249, 0.342, 0.192],
                        [0.826, 0.288, 0.288],
                        [0.850, 0.649, 0.263],
                        [0.273, 0.750, 0.230],
                        [0.320, 0.186, 0.643],
                        [0.677, 0.305, 0.683],
                        [0.788, 0.693, 0.644],
                        [0.165, 0.745, 0.702],
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1]],
                    dtype=bm.float64)

        cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                        [0, 3, 2, 1, 8, 11, 10, 9],
                        [4, 5, 6, 7, 12, 13, 14, 15],
                        [3, 7, 6, 2, 11, 15, 14, 10],
                        [0, 1, 5, 4, 8, 9, 13, 12],
                        [1, 2, 6, 5, 9, 10, 14, 13],
                        [0, 4, 7, 3, 8, 12, 15, 11]],
                        dtype=bm.int32)
        mesh = HexahedronMesh(node, cell)

        return mesh
    
