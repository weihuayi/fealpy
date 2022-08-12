import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC 

from fealpy.pde.helmholtz_2d import HelmholtzData2d
from fealpy.pde.helmholtz_3d import HelmholtzData3d

import sys



n = int(sys.argv[1])

pde = HelmholtzData2d(k=1) 

pde.symbolic_com()

domain = pde.domain()

mesh = MF.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')

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

print("残量：", np.linalg.norm(np.abs(A@uh-F)))




uI = space.interpolation(pde.solution)

e = space.integralalg.error(pde.solution, uI)
print("interpolation L2:", e)

bc = np.array([1/3, 1/3, 1/3])
uI = uI(bc)
uI0 = np.real(uI)
uI1 = np.imag(uI)

uh = uh(bc)
uh0 = np.real(uh)
uh1 = np.imag(uh)

print('real:', np.max(np.abs(uI0 - uh0)))
print('imag:', np.max(np.abs(uI1 - uh1)))

e = space.integralalg.error(pde.solution, uh)
print(" fem L2:", e)



fig, axes = plt.subplots(2, 2)
mesh.add_plot(axes[0, 0], cellcolor=uI0, linewidths=0)
mesh.add_plot(axes[0, 1], cellcolor=uI1, linewidths=0) 
mesh.add_plot(axes[1, 0], cellcolor=uh0, linewidths=0)
mesh.add_plot(axes[1, 1], cellcolor=uh1, linewidths=0) 

plt.show()
