import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.mesh_quality import RadiusRatioQuality

import MeshExample2d
from optimizer import *

mesh = MeshExample2d.triangle_domain()
node = mesh.entity('node')
mesh_quality = RadiusRatioQuality(mesh)
q = mesh_quality(node)
show_mesh_quality(q,ylim = 1000)

mesh = iterate_solver(mesh)

node = mesh.entity('node')
q = mesh_quality(node)
show_mesh_quality(q,ylim=1000)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
