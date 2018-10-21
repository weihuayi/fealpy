import sys

import numpy as np  
import matplotlib.pyplot as plt

from fealpy.pde.poisson_model_2d import CosCosData
from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.tools.show import showmultirate

n = int(sys.argv[1])

pde = CosCosData()
mesh = pde.init_mesh(n=n, meshtype='tri')
maxit = 4

Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u - u_h||_0$ with p=1',
             '$|| u - u_h||_0$ with p=2',
             '$|| u - u_h||_0$ with p=5',
             '$|| u - u_h||_0$ with p=10',
             '$||\\nabla u - \\nabla u_h||_0$ with p=1',
             '$||\\nabla u - \\nabla u_h||_0$ with p=2',
             '$||\\nabla u - \\nabla u_h||_0$ with p=5',
             '$||\\nabla u - \\nabla u_h||_0$ with p=10',
             ]

ps = [1, 2, 5, 10]
q  = [6, 8, 9, 10]

errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    Ndof[i] = mesh.number_of_nodes()
    for j, p in enumerate(ps):
        integrator = mesh.integrator(q[j])
        fem = PoissonFEMModel(pde, mesh, p, integrator)
        print(fem.get_right_vector())
        fem.solve()
        Ndof[i] = fem.femspace.number_of_global_dofs()
        errorMatrix[j, i] = fem.get_L2_error()
        errorMatrix[j+4, i] = fem.get_H1_error()
    if i < maxit - 1:
        mesh.uniform_refine()
print('Ndof',Ndof)
print(errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()

