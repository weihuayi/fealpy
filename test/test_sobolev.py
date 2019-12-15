import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.pde.sobolev_equation_2d import SinSinExpData
from fealpy.functionspace import WeakGalerkinSpace2d

nu = 1
epsilon = 0.1
pde = SinSinExpData(nu, epsilon)
domain = pde.domain()


p = 1
h = 0.1
maxit = 4
error = np.zeros((4, maxit), dtype=np.float)
for i in range(4):
    mesh = triangle(domain, h, meshtype='polygon')
    space = WeakGalerkinSpace2d(mesh, p=p)

    uh = space.projection(pde.init_value)
    gh = space.projection(lambda x:pde.gradient(x, 0.0), dim=2)

    ph = space.weak_grad(uh)
    dh = space.weak_div(gh)
    error[0, i] = space.integralalg.L2_error(pde.init_value, uh)
    error[1, i] = space.integralalg.L2_error(lambda x: pde.gradient(x, 0.0), ph)
    error[2, i] = space.integralalg.L2_error(lambda x: pde.gradient(x, 0.0), gh)
    error[3, i] = space.integralalg.L2_error(lambda x: pde.laplace(x, 0.0), dh)
    h /= 2

print(error)
print(error[:, 0:-1]/error[:, 1:])

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
