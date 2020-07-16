#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory

mf = MeshFactory()
mesh = mf.polygon_mesh(meshtype='triquad')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, showindex=True)
mesh.find_node(axes, showindex=True)

mesh.uniform_refine(n=3)

node = mesh.entity('node')
edge = mesh.entity('edge')
cell, cellLocation = mesh.entity('cell')
halfedge = mesh.entity('halfedge')


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
