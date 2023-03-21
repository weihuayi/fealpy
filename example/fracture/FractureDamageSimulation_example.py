import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh

mesh = TriangleMesh.from_square_domain_with_fracture()

mesh.uniform_refine(4)

fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
plt.show()
