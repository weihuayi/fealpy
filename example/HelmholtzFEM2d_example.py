import numpy as np
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC 

from fealpy.pde.helmholtz_3d import HelmholtzData3d
from fealpy.pde.helmholtz_2d import HelmholtzData2d



pde = HelmholtzData2d() 

domain = pde.domain()
mesh = MF.boxmesh2d(domain, nx=100, ny=100, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=1)

S = space.stiff_matrix()
M = space.mass_matrix()
P = space.penalty_matrix()

F = space.source_vector(pde.source)


A = S -  pde.k**2*M + complex(-0.07, 0.01)*P

bc = RobinBC(space, pde.robin)
A, F = bc.apply(A, F)

print(A.dtype)
print(F.dtype)


uI = space.interpolation(pde.solution)

bc = np.array([1/3, 1/3, 1/3])
u = uI(bc)

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=np.real(u))
mesh.add_plot(axes[1], cellcolor=np.imag(u)) 

fig, axes = plt.subplots(1, 2, subplot_kw={'projection':'3d'})
plt.show()
