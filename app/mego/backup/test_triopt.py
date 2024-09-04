import numpy as np
from meshpy.triangle import MeshInfo, build
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh 
from scipy.optimize import minimize

from TriRadiusRatioQuality import TriRadiusRatioQuality

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
quality = TriRadiusRatioQuality(tmesh)

x0 = quality.get_init_value()
R = minimize(quality, x0, method='Powell',  callback=quality.callback)
print(R)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')

quality.update_mesh_node(R.x)
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')

plt.show()
