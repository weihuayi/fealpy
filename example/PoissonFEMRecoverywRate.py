import sys
import numpy as np  
import matplotlib.pyplot as plt

from fealpy.model.poisson_model_2d import CosCosData

from fealpy.femmodel.PoissonFEMModel import PoissonFEMModel
from fealpy.tools.show import showmultirate
from fealpy.recovery import FEMFunctionRecoveryAlg

m = int(sys.argv[1])
n = int(sys.argv[2])
meshtype = int(sys.arav[3])

if m == 1:
    model = CosCosData()
elif m == 2:
    pass
elif m == 3:
    pass

if meshtype == 0:
    mesh = model.init_mesh(n=n, meshtype='tri')
elif meshtype == 1:
    pass


mesh.add_plot(plt)
fem = PoissonFEMModel(mesh, model, 1)

ralg = FEMFunctionRecoveryAlg()
maxit = 4

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$', 
             '$||\\nabla u - G(\\nabla u_h)||_{0}$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit):
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_cells() 
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    rguh = ralg.simple_average(uh)
    errorMatrix[3, i] = fem.recover_error(rguh)
    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix[:4], errorType[:4])
plt.show()
