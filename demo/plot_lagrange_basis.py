import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.pde.poisson_1d import CosData

n = 100
p = 4

pde = CosData()
mesh = pde.init_mesh(n=0)

bc = np.zeros((n, 2), dtype=np.float)
bc[:, 1] =  np.linspace(0, 1, n)
bc[:, 0] = 1 - bc[:, 1]

space = LagrangeFiniteElementSpace(mesh, p=p)

val = space.basis(bc)
print(val.shape)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
axes.plot(bc[:, 1], val)
axes.set_axis_on()
plt.show()
