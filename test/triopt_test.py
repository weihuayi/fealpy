
import sys

import numpy as np

import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.meshquality import TriRadiusRatio

def mesh_2dpy(mesh_info,h):
    mesh = build(mesh_info,max_volume=(h)**2)
    point = np.array(mesh.points,dtype=np.float)
    cell = np.array(mesh.elements,dtype=np.int)
    tmesh =TriangleMesh(point,cell)
    return tmesh

mesh_info = MeshInfo()  

mesh_info.set_points([(0,0), (1,0), (1,1), (0,1)])
mesh_info.set_facets([[0,1], [1,2], [2,3], [3,0]])
h = 0.05
tmesh = mesh_2dpy(mesh_info,h)

N = tmesh.number_of_points()
point = tmesh.point

quality= TriRadiusRatio()

q = quality(tmesh)

F, A, B = quality.objective_function(tmesh)

gradF = np.zeros((N, 2), dtype=np.float)
gradF[:, 0] = A@point[:, 0] + B@point[:, 1]
gradF[:, 1] = point[:, 0]@B + A@point[:, 1]
gradF = gradF/A.diagonal().reshape(-1, 1)

hh = np.sqrt(np.sum((gradF)**2, axis=1))
print(hh)


fig = plt.figure()
axes = fig.gca()
#tmesh.add_plot(axes, cellcolor=1/q, showcolorbar=True)
tmesh.add_plot(axes)
point = tmesh.point
axes.quiver(point[:, 0], point[:, 1], gradF[:, 0], gradF[:, 1])

fig = plt.figure()
axes = fig.gca()
quality.show_quality(axes, 1/q)

plt.show()
