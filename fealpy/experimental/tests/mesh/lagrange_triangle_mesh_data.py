import numpy as np
from fealpy.geometry import SphereSurface
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh

# 定义多个典型的 LagrangeTriangleMesh 对象
surface = SphereSurface() #以原点为球心，1 为半径的球
mesh = TriangleMesh.from_unit_sphere_surface()

from_triangle_mesh_data = [
        {
            "p": 3,
            "mesh": mesh,
            "surface": surface,
            "face2cell": np.array([[ 7,  8,  0,  2], [ 0,  7,  0,  1], [ 6,  7,  1,  2],
                [ 1,  5,  2,  1], [11, 12,  1,  2], [ 2, 12,  2,  0],
                [12, 13,  1,  2], [ 0,  4,  1,  2], [ 0,  1,  2,  0],
                [ 1,  2,  1,  0], [ 2,  3,  1,  2], [ 4, 15,  0,  1],
                [ 3, 18,  1,  2], [ 3,  4,  0,  1], [ 6, 10,  0,  1],
                [ 5,  6,  0,  2], [ 5, 11,  2,  0], [11, 17,  2,  2],
                [ 9, 10,  2,  0], [16, 17,  1,  0], [10, 17,  2,  1],
                [ 8, 15,  1,  2], [ 8,  9,  0,  1], [14, 15,  2,  0],
                [ 9, 19,  0,  2], [13, 16,  1,  2], [13, 18,  0,  1],
                [14, 18,  1,  0], [16, 19,  0,  0], [14, 19,  0,  1]], dtype=np.int32),
            "NN": 92,
            "NE": 30,
            "NC": 20
                }
]
