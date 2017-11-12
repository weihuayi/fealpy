
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.geometry.gmsh_geo import GmshGeo
from fealpy.geometry.gmsh_mesh import generate_mesh
from CGAL.CGAL_Kernel import Point_2, Segment_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
import meshio


def generate_random_segments(box, num):
    o = np.array([box[0], box[2]])
    l = np.array([box[1] - box[0], box[3] - box[2]])
    p0 = o + l*np.random.rand(num, 2)
    p1 = o + l*np.random.rand(num, 2)
    l = np.sqrt(np.sum((p0-p1)**2, axis=1))
    print(l)
    return p0[l>1e-3], p1[l>1e-3] 

def get_sub_segments(p0, p1):
    cdt = Constrained_Delaunay_triangulation_2()
    for s0, s1 in zip(p0, p1):
        cdt.insert_constraint(Point_2(s0[0], s0[1]), Point_2(s1[0], s1[1]))
        
    d = {v:i for i, v in enumerate(cdt.finite_vertices())}
    subsegments = []
    for edge in cdt.finite_edges():
        if cdt.is_constrained(edge):
            v0 = edge[0].vertex(cdt.ccw(edge[1]))
            v1 = edge[0].vertex(cdt.cw(edge[1]))
            subsegments.append((d[v0], d[v1]))

    points = [(p.x(), p.y()) for p in cdt.points()]

    return points, subsegments

box = [0.0, 1.0, 0.0, 1.0]
geom = GmshGeo() 
cl = 0.1
poly = geom.add_rectangle(box, cl)

p0, p1 = generate_random_segments(box, 5)
ps, subsegs = get_sub_segments(p0, p1)
pp = geom.add_points(ps, 0.02)
lines = geom.add_segments_in_surface(pp, subsegs, poly.surface)
geom.add_physical_point(pp)
geom.add_physical_line(lines)
geom.add_physical_surface([poly.surface])

#f = geom.add_boundarylayer_field(pp, lines, 
#        hfar=cl, 
#        hwall_n=0.02, 
#        ratio=1.1,
#        thickness=0.1)
#geom.set_background_field(f)
geom.set_mesh_algorithm(2)

point, cell, point_data, cell_data, field_data = generate_mesh(geom,
        optimize=False)
print(cell)
print(point_data)
print(cell_data)
print(field_data)
tmesh = TriangleMesh(point[:, 0:2], cell['triangle'])
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()

