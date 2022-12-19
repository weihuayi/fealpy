import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh.quality import InverseMeanRatio, TriRadiusRatio

node = np.array([
    (0, 2), (1.73205080, 1), (1.73205080, -1), (.8, -1.5), 
    (-1.73205080, -1), (-1.73205080, 1), (-0.8, -0.8), (1.73205080, -3),
    (-1.73205080, -3), (0, -4)], dtype=np.float64) 

cell = np.array([
    (6, 1, 0),
    (6, 2, 1),
    (6, 3, 2),
    (6, 4, 3),
    (6, 5, 4),
    (6, 0, 5),
    (3, 7, 2),
    (3, 9, 7),
    (3, 8, 9),
    (3, 4, 8)], dtype=np.int32)

mesh = TriangleMesh(node, cell)

w = np.array([(1, 0.5), (0, np.sqrt(3)/2)], dtype=np.float64)

q0 = InverseMeanRatio(w)

q = q0.quality(mesh)
q0.show(q)

q1 = TriRadiusRatio()
q = q1.quality(mesh)
q1.show(q)

fig = plt.figure()
axes  = fig.gca()
mesh.add_plot(axes)
plt.show()


