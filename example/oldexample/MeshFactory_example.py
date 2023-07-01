#!/usr/bin/env python3
# 
import sys 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import MeshFactory
from fealpy.geometry import CircleCurve, FoldCurve

mf = MeshFactory


# 2d mesh
box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx=4, ny=4, meshtype='tri')

mesh = mf.boxmesh2d(box, nx=4, ny=4, meshtype='quad')

mesh = mf.boxmesh2d(box, nx=4, ny=4, meshtype='poly')

mesh = mf.triangle(box, h=0.1, meshtype='tri')

mesh = mf.triangle(box, h=0.1, meshtype='poly')

mesh = mf.special_boxmesh2d(box, n=10, meshtype='fishbone')

mesh = mf.special_boxmesh2d(box, n=10, meshtype='rice')

mesh = mf.special_boxmesh2d(box, n=10, meshtype='cross')

mesh = mf.special_boxmesh2d(box, n=10, meshtype='nonuniform')

mesh = mf.unitcirclemesh(0.1, meshtype='tri')

mesh = mf.unitcirclemesh(0.1, meshtype='poly')


from fealpy.mesh import MeshFactory
from fealpy.geometry import CircleCurve, FoldCurve
interface = FoldCurve(a=6)
mesh = mf.interfacemesh2d(interface, n=40)
fig = plt.figure()
axes = fig.gca() 
mesh.add_plot(axes)
plt.show()


# 3d mesh
if False:
    box = [0, 1, 0, 1, 0, 1]
    mesh = mf.boxmesh3d(box, nx=1, ny=1, nz=1, meshtype='hex')

    mesh = mf.boxmesh3d(box, nx=1, ny=1, nz=1, meshtype='tet')



# plot 
GD = mesh.geo_dimension()
fig = plt.figure()
if GD == 2:
    axes = fig.gca() 
    mesh.add_plot(axes)
#    mesh.find_node(axes, showindex=True)
#    mesh.find_cell(axes, showindex=True)
elif GD == 3:
    axes = fig.gca(projection='3d')
    mesh.add_plot(axes)

plt.show()
