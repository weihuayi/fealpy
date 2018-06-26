import numpy as np
from meshpy.triangle import MeshInfo, build
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 
from scipy.optimize import minimize

from TriRadiusRatio import TriRadiusRatio

h = 0.05
mesh_info = MeshInfo()

# Set the vertices of the domain [0, 1]^2
mesh_info.set_points([
    (0,0), (1,0), (1,1), (0,1)])

# Set the facets of the domain [0, 1]^2
mesh_info.set_facets([
    [0,1],
    [1,2],
    [2,3],
    [3,0]
    ])

# Generate the tet mesh
mesh = build(mesh_info, max_volume=(h)**2)
node = np.array(mesh.points, dtype=np.float)
cell = np.array(mesh.elements, dtype=np.int)
tmesh = TriangleMesh(node, cell)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')

trirr = TriRadiusRatio(tmesh)
q = trirr.get_quality()
print('q=',max(q))
trirr.iterate_solver()
q = trirr.get_quality()
print('q=',max(q))

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')
#axes.quiver(tmesh.node[:, 0], tmesh.node[:, 1], p[:, 0]/NN, p[:, 1]/NN)
plt.show()
