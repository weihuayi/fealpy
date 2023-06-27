import numpy as np
import sys
import meshio
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 


point, cell, _, _, _ = meshio.read(sys.argv[1])

tmesh = TriangleMesh(point[:, 0:2], cell['triangle'])
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()
