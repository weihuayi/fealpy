import sys

import numpy as np

import matplotlib.pyplot as plt


from fealpy.mesh.interface_mesh_generator import InterfaceMesh2d

from fealpy.mesh.PolygonMesh import PolygonMesh 
from fealpy.mesh.vtkMeshIO import write_vtk_mesh 

from fealpy.mesh.curve import CircleCurve
from fealpy.mesh.curve import Curve1
from fealpy.mesh.curve import Curve2 
from fealpy.mesh.curve import Curve3

import time

n = 40

interface = Curve1(a=6)
#interface = Curve2()
#interface = Curve3()

t0  = time.clock()
alg = InterfaceMesh2d(interface, interface.box, n)
ppoint, pcell, pcellLocation = alg.run()
t1 = time.clock()
pointMarker = alg.point_marker()
pmesh = PolygonMesh(ppoint, pcell, pcellLocation)
a = pmesh.angle()
print(a.max())
write_vtk_mesh(pmesh, 'test.vtk')

fig = plt.figure()
axes = fig.gca() 
pmesh.add_plot(axes, cellcolor=[0.5, 0.9, 0.45])
pmesh.find_point(axes, point=pmesh.point[pointMarker], markersize=30)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.savefig('sixfolds.pdf')
plt.show()
