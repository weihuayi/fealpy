#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadtreeMesh, QuadtreeForest

maxdepth = 4
node = np.array([
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1)], dtype=np.float)

cell = np.array([(0, 1, 2, 3)], dtype=np.int)

mesh = QuadtreeMesh(node, cell)

forest = QuadtreeForest(mesh, maxdepth)
forest.uniform_refine(n=1)
forest.octant()
forest.print()

forest.add_plot(plt)
plt.show()

