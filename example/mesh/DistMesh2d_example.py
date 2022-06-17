#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TriMesher import distmesh 

from fealpy.geometry import dcircle, drectangle
from fealpy.geometry import ddiff, huniform



h = 0.05
fd = lambda p: drectangle(p, [0.0, 1.0, 0.0, 1.0])
fh = huniform
bbox = [-0.2, 1.2, -0.2, 1.2]
pfix = np.array([
    (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float64)
mesh = distmesh(h, fd, fh, bbox, pfix, showanimation=True)

fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

