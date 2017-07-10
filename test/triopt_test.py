
import sys

import numpy as np

import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.meshquality import TriRadiusRatio
from fealpy.mesh.meshopt import OptAlg

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

quality = TriRadiusRatio()
optalg = OptAlg(tmesh, quality)
optalg.run(maxit=20)

fig = plt.figure()
axes = fig.gca()
#tmesh.add_plot(axes, cellcolor=1/q, showcolorbar=True)
tmesh.add_plot(axes)
point = tmesh.point
#axes.quiver(point[:, 0], point[:, 1], gradF[:, 0], gradF[:, 1])

fig = plt.figure()
axes = fig.gca()
q = quality(tmesh.point, tmesh.ds.cell)
quality.show_quality(axes, 1/q)

plt.show()
