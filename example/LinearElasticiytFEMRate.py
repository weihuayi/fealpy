import numpy as np
import sys

from fealpy.model.linear_elasticity_model import PolyModel3d, Model2d
from fealpy.femmodel.LinearElasticityFEMModel import LinearElasticityFEMModel 
from fealpy.tools.show import showmultirate

import numpy as np  
import matplotlib.pyplot as plt


m = int(sys.argv[1])
p = int(sys.argv[2])
n = int(sys.argv[3])

if m == 1:
    model = PolyModel3d()
if m == 2:
    model = Model2d()

mesh = model.init_mesh(n)
integrator = mesh.integrator(10)

fem = LinearElasticityFEMModel(mesh, model, p, integrator)

maxit = 4 

errorType = ['$||\sigma - \sigma_h ||_{0}$',
             '$||div(\sigma - \sigma_h)||_{0}$',
             '$||u - u_h||_{0}$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    fem.solve()
    Ndof[i] = len(fem.sh)
    e0, e1, e2 = fem.error()
    errorMatrix[0, i] = e0
    errorMatrix[1, i] = e1 
    errorMatrix[2, i] = e2 
    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)
        

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
