import sys

import numpy as np  
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.tools.show import showmultirate

p = int(sys.argv[1])
q = int(sys.argv[2])
n = int(sys.argv[3])

pde = CosCosData()
mesh = pde.init_mesh(n=n, meshtype='tri')
integrator = mesh.integrator(q)
maxit = 4

errorType = ['$|| u - u_h||_{0}$', '$||\\nabla u - \\nabla u_h||_{0}$']
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit):
    fem = PoissonFEMModel(pde, mesh, p, integrator)
    fem.solve()
    Ndof[i] = fem.femspace.number_of_global_dofs()
    errorMatrix[0, i] = fem.get_L2_error()
    errorMatrix[1, i] = fem.get_H1_error()
    if i < maxit - 1:
        mesh.uniform_refine()

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix[:3, :],  errorType[:3])
plt.show()

