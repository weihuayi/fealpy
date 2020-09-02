
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace


box = [0, 1, 0, 1]
mf = MeshFactory()
mesh = mf.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=3)
ps = space.interpolation_points()

P = space.linear_interpolation_matrix()
print('array:', P.toarray())

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ps, showindex=True)
plt.show()



