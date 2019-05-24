#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadtreeMesh

b = 5
node = np.array([
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1)], dtype=np.float)

cell = np.array([(0, 1, 2, 3)], dtype=np.int)

mesh = QuadtreeMesh(node, cell)
mesh.uniform_refine(b)
mesh.node *= 2**b
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, showaxis=True, cellcolor='lightgray', edgecolor='gray',
        linewidths=1)
plt.xticks(np.arange(2**b+1))
plt.yticks(np.arange(2**b+1))
plt.show()

