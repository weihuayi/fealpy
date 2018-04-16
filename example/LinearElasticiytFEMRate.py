import numpy as np
import sys

from fealpy.model.linear_elasticity_model import PolyModel3d, Model2d, SimplifyModel2d, HuangModel2d
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
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
if m == 3:
    model = SimplifyModel2d()
if m == 4:
    model = HuangModel2d()

#box = [0, 1, 0, 1]
#mesh = rectangledomainmesh(box, nx=n, ny=n)

mesh = model.init_mesh(n)
integrator = mesh.integrator(7)


maxit = 4 

errorType = ['$||\sigma - \sigma_h ||_{0}$',
             '$||div(\sigma - \sigma_h)||_{0}$',
             '$||u - u_h||_{0}$',
             '$||\sigma - \sigma_I ||_{0}$',
             '$||div(\sigma - \sigma_I)||_{0}$'
             ]
Ndof = np.zeros((maxit,))
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    fem = LinearElasticityFEMModel(mesh, model, p, integrator)
    fem.solve()
    
#    fig = plt.figure()
#    axes = fig.gca()
#    mesh.print()
#    mesh.add_plot(axes)
#    mesh.find_node(axes, showindex=True)
#    mesh.find_edge(axes, showindex=True)
#    mesh.find_cell(axes, showindex=True)

    Ndof[i] = fem.mesh.number_of_cells() 
    errorMatrix[:, i] = fem.error()
    if i < maxit - 1:
        #n *= 2
        #mesh = rectangledomainmesh(box, nx=n, ny=n)
        mesh.uniform_refine()
        

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 1, Ndof, errorMatrix, errorType)
plt.show()
