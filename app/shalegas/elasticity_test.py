#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory


mfactory = MeshFactory()

box = [0, 1, 0, 1]

mesh = mfactory.regular(box)
node = mesh.entity('node')
edge = mesh.entity('edge')
bc = mesh.entity_barycenter('edge')
isFractureEdge = (bc[:, 1] == 0.5) & (bc[:, 0] > 0.2) & (bc[:, 0] < 0.8)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_edge(axes, index=isFractureEdge)
plt.show()




