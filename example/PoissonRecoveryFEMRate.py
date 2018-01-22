
import sys

import numpy as np  
import matplotlib.pyplot as plt
from fealpy.model.poisson_model_2d import CosCosData
from fealpy.femmodel.PoissonRecoveryFEMModel import PoissonRecoveryFEMModel
from fealpy.tools.show import showmultirate

from fealpy.functionspace import FunctionNorm
from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 

m = int(sys.argv[1])
n = int(sys.argv[2])

if m == 1:
    model = CosCosData()

mesh = model.init_mesh(n=n)
integrator = TriangleQuadrature(3)
fem = PoissonRecoveryFEMModel(mesh, model, integrator, rtype='harmonic')
funNorm = FunctionNorm(integrator, fem.area)
maxit = 4

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit):
    fem.solve()
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = funNorm.l2_error(model.solution, fem.uh)
    errorMatrix[1, i] = funNorm.L2_error(model.solution, fem.uh)
    errorMatrix[2, i] = funNorm.H1_semi_error(model.gradient, fem.uh)
    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)
        funNorm.area = fem.area

print('Ndof:', Ndof)
print('error:', errorMatrix)
fig = plt.figure()
fig.set_facecolor('white')
axes = fig.gca()
optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, 0, Ndof, errorMatrix[:3, :], optionlist[:3], errorType[:3])
axes.legend(loc=3, prop={'size': 30})
plt.show()

