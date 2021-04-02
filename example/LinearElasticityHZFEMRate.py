import numpy as np
import sys

from fealpy.pde.linear_elasticity_model import QiModel3d, PolyModel3d, Model2d, HuangModel2d
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
from fealpy.fem.LinearElasticityFEMModel import LinearElasticityFEMModel 
from fealpy.tools.show import showmultirate

import numpy as np  
import matplotlib.pyplot as plt


m = int(sys.argv[1])
p = int(sys.argv[2])
n = int(sys.argv[3])

if m == 1:
    pde = PolyModel3d()
    #pde = QiModel3d()
if m == 2:
    pde = Model2d()
if m == 3:
    pde = SimplifyModel2d()
if m == 4:
    pde = HuangModel2d()

#box = [0, 1, 0, 1]
#mesh = rectangledomainmesh(box, nx=n, ny=n)

mesh = pde.init_mesh(n)
integrator = mesh.integrator(5)

maxit = 3 

errorType = ['$||\sigma - \sigma_h ||_{0}$',
             '$||div(\sigma - \sigma_h)||_{0}$',
             '$||u - u_h||_{0}$',
             '$||\sigma - \sigma_I ||_{0}$',
             '$||div(\sigma - \sigma_I)||_{0}$',
             '$||u - u_I||_{0}$',
             '$|| u_I - u_h||_{0}$'
             ]
Ndof = np.zeros((maxit,))
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    fem = LinearElasticityFEMModel(mesh, pde, p, integrator)
    fem.solve()
    #fem.fast_solve()
    tgdof = fem.tensorspace.number_of_global_dofs()
    vgdof = fem.vectorspace.number_of_global_dofs()
    gdof = tgdof + vgdof
    Ndof[i] = gdof 
    errorMatrix[:, i] = fem.error()
    if i < maxit - 1:
        mesh.uniform_refine()
        
print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
