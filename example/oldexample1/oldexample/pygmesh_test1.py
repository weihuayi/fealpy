import pygmsh

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 

geom = pygmsh.opencascade.Geometry(
  characteristic_length_min=0.1,
  characteristic_length_max=0.1,
  )

rectangle = geom.add_rectangle([-1.0, -1.0, 0.0], 2.0, 2.0)
disk1 = geom.add_disk([-1.2, 0.0, 0.0], 0.5)
disk2 = geom.add_disk([+1.2, 0.0, 0.0], 0.5)
union = geom.boolean_union([rectangle, disk1, disk2])

disk3 = geom.add_disk([0.0, -0.9, 0.0], 0.5)
disk4 = geom.add_disk([0.0, +0.9, 0.0], 0.5)
flat = geom.boolean_difference([union], [disk3, disk4])

#geom.extrude(flat, [0, 0, 0.3])

point, cell, point_data, cell_data, field_data = pygmsh.generate_mesh(geom)

tmesh = TriangleMesh(point[:, 0:2], cell['triangle'])
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()
