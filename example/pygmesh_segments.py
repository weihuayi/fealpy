import pygmsh
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 
import meshio

geom = pygmsh.built_in.Geometry()

poly = geom.add_polygon([
    [0.0,   0.0, 0.0],
    [1.0,   0.0, 0.0],
    [1.0,   1.0, 0.0],
    [0.0,   1.0, 0.0],],
    lcar=0.01
    )
p0 = geom.add_point([0.1, 0.2, 0.0], 0.01)
p1 = geom.add_point([0.8, 0.9, 0.0], 0.01)
line0 = geom.add_line(p0, p1)
geom.add_line_in_surface(line0, poly.surface)
geom.set_mesh_algorithm(1)

#axis = [0, 0, 1]

#geom.extrude(
#    poly,
#    translation_axis=axis,
#    rotation_axis=axis,
#    point_on_axis=[0, 0, 0],
#    angle=2.0 / 6.0 * np.pi
#    )

point, cell, point_data, cell_data, field_data = pygmsh.generate_mesh(geom, optimize=False)

#point, cell, pt_data, cell_data, field_data = meshio.read('out.msh')
tmesh = TriangleMesh(point[:, 0:2], cell['triangle'])
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()

