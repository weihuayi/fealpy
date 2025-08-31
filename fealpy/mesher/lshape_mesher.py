from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh, QuadrangleMesh

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

    @init_mesh.register('tri_threshold')
    def init_mesh(self,
                  big_box:list[float]=(-1, 1, -1, 1),
                  small_box:list[float]=(0, 1, 0, 1),
                  nx: int = 10, ny: int = 10) -> TriangleMesh:
        """
        Create a mesh for an L-shaped domain with a specified big box and small box.

        Parameters
            big_box: List[float]
                The coordinates of the big box in the format [xmin, xmax, ymin, ymax].
            small_box: List[float]
                The coordinates of the small box in the format [xmin, xmax, ymin, ymax].
            nx: int
                Number of divisions along the x-axis.
            ny: int
                Number of divisions along the y-axis.
        Returns
            TriangleMesh
                A mesh object representing the L-shaped domain.
        """

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x>=small_box[0])
                   &(x<=small_box[1])
                   &(y>=small_box[2])
                   &(y<=small_box[3]))

        l_shape_mesh = TriangleMesh.from_box(big_box,
                                             nx=nx, ny=ny,
                                             threshold=threshold)

        return l_shape_mesh


    @init_mesh.register('quad')
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
            [0, 1, 3, 2],
            [3, 4, 7, 6],
            [3, 6, 5, 2]], dtype=bm.int32)
        mesh = QuadrangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

    @init_mesh.register('quad_threshold')
    def init_mesh(self,
                  big_box:list[float]=(-1, 1, -1, 1),
                  small_box:list[float]=(0, 1, 0, 1),
                  nx: int = 10, ny: int = 10) -> QuadrangleMesh:
        """
        Create a mesh for an L-shaped domain with a specified big box and small box.

        Parameters
            big_box: List[float]
                The coordinates of the big box in the format [xmin, xmax, ymin, ymax].
            small_box: List[float]
                The coordinates of the small box in the format [xmin, xmax, ymin, ymax].
            nx: int
                Number of divisions along the x-axis.
            ny: int
                Number of divisions along the y-axis.
        Returns
            QuadrangleMesh
                A mesh object representing the L-shaped domain.
        """

        def threshold(p):
            x = p[..., 0]
            y = p[..., 1]
            return ((x>=small_box[0])
                   &(x<=small_box[1])
                   &(y>=small_box[2])
                   &(y<=small_box[3]))

        l_shape_mesh = QuadrangleMesh.from_box(big_box,
                                             nx=nx, ny=ny,
                                             threshold=threshold)
        return l_shape_mesh