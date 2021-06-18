

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.functionspace import LagrangeFiniteElementSpace


pde = PDE()

mesh = pde.init_mesh(n=4, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, 2)

uI = space.interpolation(pde.solution) # 有限元空间的函数，同地它也是一个数组

bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
val = uI(bc) # (NC, )
val = uI.value(bc) # (NC, )
gval = uI.grad_value(bc) #(NC, GD)

L2error = space.integralalg.error(pde.solution, uI, q=5, power=2)
H1error = space.integralalg.error(pde.gradient, uI.grad_value, q=5, power=2)
print(H1error)

fig = plt.figure()
axes = fig.add_subplot(projection='3d')
uI.add_plot(axes, cmap='rainbow')
plt.show()
