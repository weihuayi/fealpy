
import numpy as np
import matplotlib.pyplot as plt
import sys

from fealpy.mesh.StructureQuadMesh import StructureQuadMesh 
box = [0, 1, 0, 1]
n = 2
qmesh = StructureQuadMesh(box, n, n)
qmesh.print()

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes)
qmesh.find_point(axes, showindex=True)
qmesh.find_edge(axes, showindex=True)
qmesh.find_cell(axes, showindex=True)
plt.show()
