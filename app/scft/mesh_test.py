
import numpy as np
import matplotlib.pyplot as plt

from vem2d_problem import halfedgemesh, init_mesh

quadtree = init_mesh(n=2, h=12)

mesh0 = quadtree.to_pmesh()
node0 = mesh0.entity('node')
print(node0)
print(node0.dtype)

mesh1 = halfedgemesh(n=2, h=12)
node1 = mesh1.entity('node')
print(node1)
print(node1.dtype)


fig = plt.figure()
axes = fig.gca()
mesh0.add_plot(axes)

fig = plt.figure()
axes = fig.gca()
mesh1.add_plot(axes)
plt.show()
