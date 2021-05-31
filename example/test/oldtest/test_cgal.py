
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh 
from CGAL.CGAL_Kernel import Point_2, Segment_2
from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
import meshio

def insert_segments(cdt, segs):
    '''Insert some segments into cdt, here the segs maybe instected
    '''
    pass


p0 = Point_2(0.1, 0.1)
p1 = Point_2(0.8, 0.9)
p2 = Point_2(0.1, 0.8)
p3 = Point_2(0.8, 0.2)
cdt = Constrained_Delaunay_triangulation_2()

cdt.insert_constraint(p0, p1)
cdt.insert_constraint(p2, p3)
d = {j:i for i, j in enumerate(cdt.finite_vertices())}
vs = [v for v in cdt.finite_vertices()]

for edge in cdt.finite_edges():
    if cdt.is_constrained(edge):
        v0 = edge[0].vertex(cdt.ccw(edge[1]))
        v1 = edge[0].vertex(cdt.cw(edge[1]))
        print((d[v0], d[v1]))


