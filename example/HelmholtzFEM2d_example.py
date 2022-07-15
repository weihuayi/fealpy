import numpy as np
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.pde.helmholtz_3d import HelmholtzData3d
from fealpy.pde.helmholtz_2d import HelmholtzData2d



pde = HelmholtzData2d() 
domain = pde.domain()
mesh = MF.boxmesh2d(domain, nx=2, ny=2, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=1)

S = space.stiff_matrix()
P = space.penalty_matrix()

A = S + complex(-0.07, 0.01)*P


uI = space.interpolation(pde.solution)

bc = np.array([1/3, 1/3, 1/3])
u = uI(bc)

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=np.real(u))
mesh.add_plot(axes[1], cellcolor=np.imag(u)) 

fig, axes = plt.subplots(1, 2, subplot_kw={'projection':'3d'})
plt.show()
