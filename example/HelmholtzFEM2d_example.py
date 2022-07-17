import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC 

from fealpy.pde.helmholtz_3d import HelmholtzData3d
from fealpy.pde.helmholtz_2d import HelmholtzData2d



pde = HelmholtzData2d(k=1) 

domain = pde.domain()
mesh = MF.boxmesh2d(domain, nx=40, ny=40, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=1)

uh = space.function(dtype=np.complex128)
S = space.stiff_matrix()
M = space.mass_matrix()
P = space.penalty_matrix()

F = space.source_vector(pde.source)


# A = S -  pde.k**2*M + complex(-0.07, 0.01)*P
A = S -  pde.k**2*M 

bc = RobinBC(space, pde.robin)
A, F = bc.apply(A, F)

uh[:] = spsolve(A, F)

print(np.linalg.norm(np.abs(A@uh-F)))


print(A.dtype)
print(F.dtype)


uI = space.interpolation(pde.solution)

bc = np.array([1/3, 1/3, 1/3])
uI = uI(bc)
uI0 = np.real(uI)
uI1 = np.imag(uI)

uh = uh(bc)
uh0 = np.real(uh)
uh1 = np.imag(uh)

print('real:', np.max(np.abs(uI0 - uh0)))
print('imag:', np.max(np.abs(uI1 - uh1)))



fig, axes = plt.subplots(2, 2)
mesh.add_plot(axes[0, 0], cellcolor=uI0, linewidths=0)
mesh.add_plot(axes[0, 1], cellcolor=uI1, linewidths=0) 
mesh.add_plot(axes[1, 0], cellcolor=uh0, linewidths=0)
mesh.add_plot(axes[1, 1], cellcolor=uh1, linewidths=0) 

plt.show()
