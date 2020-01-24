#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh


node = np.array([
    (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float)
cell = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.int)

mesh = IntervalMesh(node, cell)
mesh.uniform_refine(n=4)

h = mesh.entity_measure('cell')
bc = mesh.entity_barycenter('cell')
n = mesh.cell_normal()

newNode = bc - n*np.sqrt(3)/2

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=newNode)
plt.show()


