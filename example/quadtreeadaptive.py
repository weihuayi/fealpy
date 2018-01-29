import numpy as np 
import sys
from fealpy.mesh.implicit_curve import Circle, Curve1
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.mesh.adaptive_interface_mesh_generator import AdaptiveMarker2d, QuadtreeInterfaceMesh2d 

import  matplotlib.pyplot as plt


phi = Curve1(a=12)
#phi = Curve3()
mesh = rectangledomainmesh(phi.box, nx=10, ny=10, meshtype='quad')
marker = AdaptiveMarker2d(phi, maxh=0.1, maxa=2)
alg = QuadtreeInterfaceMesh2d(mesh, marker)
pmesh= alg.get_interface_mesh()
fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
#pmesh.find_point(axes, point=cutPoint, markersize=10)
plt.show()

