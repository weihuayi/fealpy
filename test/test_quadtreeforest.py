import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadtreeMesh


node = np.array([
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1)], dtype=np.float)

cell = np.array([(0, 1, 2, 3)], dtype=np.int)

mesh = QuadtreeMesh(node, cell)
mesh.uniform_refine(5)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

