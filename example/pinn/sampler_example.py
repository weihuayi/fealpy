import numpy as np
import torch

from fealpy.pinn.sampler import TriangleMeshSampler
from fealpy.mesh import MeshFactory as Mf

mesh = Mf.boxmesh2d([0, 1, 0, 1], nx=10, ny=10)

tms = TriangleMeshSampler(10, mesh=mesh)
samples = tms.run()
print(tms.run())


### Draw the result

from matplotlib import pyplot as plt

fig, axes = plt.subplots()
mesh.add_plot(axes)
axes.scatter(samples[:, 0], samples[:, 1])
plt.show()
