
import numpy as np
from fealpy.mesh.TetMesher import meshpy3d

points = np.array([
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ], dtype=np.float64)

facets = np.array([
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
    ], dtype=np.int_)

h = 0.1

mesh = meshpy3d(points, facets, h)

mesh.to_vtk(fname='test.vtu')



