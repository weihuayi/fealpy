import sys

import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3

from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh 

p1 = np.array([0, 1], dtype=np.float)
c1 = np.array([0, 1], dtype=np.int)

p2 = np.array([
    (0, 0), 
    (1, 0), 
    (0, 1)], dtype=np.float)
c2 = np.array([
    (0, 1, 2) 
    ], dtype=np.int)

mesh2 = TriangleMesh(p2, c2)
mesh2.print()


p3 = np.array([
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1)], dtype=np.float)

c3 = np.array([
    (0, 1, 2, 3)], dtype=np.int)
mesh3 = TetrahedronMesh(p3, c3)


fig = plt.figure()
a1 = fig.add_subplot(1, 3, 1)
a2 = fig.add_subplot(1, 3, 2)
a3 = fig.add_subplot(1, 3, 3)
a1.set_title('1-simplex')
a2.set_title('2-simplex')
a3.set_title('3-simplex')

mesh2.add_plot(a2)
#mesh2.find_point(a2, showindex=True)

mesh2.add_plot(a3)
#mesh2.find_point(a3, showindex=True)

plt.show()
