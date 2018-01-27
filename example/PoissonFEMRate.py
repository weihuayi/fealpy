import sys

import numpy as np  
import matplotlib.pyplot as plt
from fealpy.model.poisson_model_2d import CosCosData
from fealpy.femmodel.PoissonFEMModel import PoissonFEMModel

from fealpy.tools.show import showmultirate

from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 

m = int(sys.argv[1])
p = int(sys.argv[2])
n = int(sys.argv[3])

if m == 1:
    model = CosCosData()

mesh = model.init_mesh(n=n, meshtype='tri')
integrator = TriangleQuadrature(3)
fem = PoissonFEMModel(mesh, model, p=p, integrator=integrator)
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
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)
        

print('Ndof:', Ndof)
print('error:', errorMatrix)
fig = plt.figure()
fig.set_facecolor('white')
axes = fig.gca()
optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, 0, Ndof, errorMatrix[:3, :], optionlist[:3], errorType[:3])
axes.legend(loc=3, prop={'size': 30})
plt.show()

