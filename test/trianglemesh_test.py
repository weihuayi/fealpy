
import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh.TriangleMesh import TriangleMesh 

point = np.array([
    (0, 0), 
    (1, 0), 
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0), 
    (3, 0, 2)], dtype=np.int)
tmesh = TriangleMesh(point, cell)
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_point(axes, showindex=True)
tmesh.find_edge(axes, showindex=True)
tmesh.find_cell(axes, showindex=True)
plt.show()
