from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh

class LshapeMesher:
    """Lshape domain mesh generator"""
    def __init__(self):
        pass

    def geo_dimension(self) -> int:
        return 2

    @variantmethod('tri')
    def init_mesh(self, n:int=2): 
        node = bm.array([
        [-1, -1],
        [ 0, -1],
        [-1,  0],
        [ 0,  0],
        [ 1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1]], dtype=bm.float64)
    
        cell = bm.array([
            [1, 3, 0],
            [2, 0, 3],
            [3, 6, 2],
            [5, 2, 6],
            [4, 7, 3],
            [6, 3, 7]], dtype=bm.int32)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh   
    
    @init_mesh.register('quad')
    def init_mesh(self, n:int=2):
        from ..mesh import QuadrangleMesh
        node = bm.array([
            [-1, -1],
            [ 0, -1],
            [-1,  0],
            [ 0,  0],
            [ 1,  0],
            [-1,  1],
            [ 0,  1],
            [ 1,  1]], dtype=bm.float64)
        cell = bm.array([
            [0, 1, 3, 2],
            [3, 4, 7, 6],
            [3, 6, 5, 2]], dtype=bm.int32)
        mesh = QuadrangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh