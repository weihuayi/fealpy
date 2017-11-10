
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.geometry.gmsh_geo import GmshGeo
from fealpy.geometry.gmsh_mesh import generate_mesh
import meshio

geom = GmshGeo() 

poly = geom.add_polygon([
    [0.0,   0.0],
    [1.0,   0.0],
    [1.0,   1.0],
    [0.0,   1.0]], 0.05
    )
p0 = geom.add_point([0.1, 0.2], 0.05)
p1 = geom.add_point([0.8, 0.9], 0.05)
line0 = geom.add_line(p0, p1)
geom.add_line_in_surface(line0, poly.surface)

p0 = geom.add_point([0.1, 0.8], 0.05)
p1 = geom.add_point([0.8, 0.2], 0.05)
line0 = geom.add_line(p0, p1)
geom.add_line_in_surface(line0, poly.surface)
geom.set_mesh_algorithm(1)

point, cell, point_data, cell_data, field_data = generate_mesh(geom)

tmesh = TriangleMesh(point[:, 0:2], cell['triangle'])
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()

