import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.mesh.simple_mesh_generator import distmesh2d 
from fealpy.mesh.level_set_function import drectangle
import triangle as tri
from scipy.spatial import Delaunay

box = [0, 1, 0, 1]
pfix = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]], dtype=np.float)
fd = lambda p: drectangle(p, box)

bbox = [-0.1, 1.1, -0.1, 1.1]
pmesh = distmesh2d(fd, 0.01, box, pfix, meshtype='polygon')

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)


node = pmesh.entity('node')
t = Delaunay(node)
tmesh = TriangleMesh(node, t.simplices.copy())

area = tmesh.entity_measure('cell')
tmesh.delete_cell(area < 1e-8)
area = tmesh.entity_measure('cell')

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()



