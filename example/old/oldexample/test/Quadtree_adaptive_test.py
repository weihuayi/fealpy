#!/usr/bin/env python3
# 
import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import Quadtree

node = np.array([
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1)], dtype=np.float)

cell = np.array([(0, 1, 3, 2)], dtype=np.int)

mesh = Quadtree(node, cell)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)


aopts = mesh.adaptive_options(method='numrefine',maxcoarsen=3,HB=True)
NC = mesh.number_of_cells()
eta = 2*np.ones(NC,dtype=int)
mesh.adaptive(eta, aopts)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()
















