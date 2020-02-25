import sys
import numpy as np  
import matplotlib.pyplot as plt
import scipy.io as sio

from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.tools.show import showmultirate, show_error_table

from fealpy.pde.poisson_2d import CosCosData as PDE 

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.mesh.simple_mesh_generator import distmesh2d 
from fealpy.mesh.level_set_function import drectangle
import triangle as tri
from scipy.spatial import Delaunay

box = [0, 1, 0, 1]
pfix = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]], dtype=np.float)
fd = lambda p: drectangle(p, box)
h=0.2
pmesh = distmesh2d(fd, h, box, pfix, meshtype='polygon')

node = pmesh.entity('node')
t = Delaunay(node)
tmesh = TriangleMesh(node, t.simplices.copy())

area = tmesh.entity_measure('cell')
tmesh.delete_cell(area < 1e-8)
area = tmesh.entity_measure('cell')
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
plt.show()
p = 1
maxit = 5

pde = PDE()
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']

errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)

integrator = tmesh.integrator(7)

for i in range(maxit):
    fem = PoissonFEMModel(pde, tmesh, p, integrator)
    ls = fem.solve()
    sio.savemat('test%d.mat'%(i), ls)
    Ndof[i] = fem.femspace.number_of_global_dofs()
    errorMatrix[0, i] = fem.get_L2_error()
    errorMatrix[1, i] = fem.get_H1_error()
    if i < maxit - 1:
        #tmesh.uniform_refine()
        h = h/2
        pmesh = distmesh2d(fd, h, box, pfix, meshtype='polygon')
        node = pmesh.entity('node')
        t = Delaunay(node)
        tmesh = TriangleMesh(node, t.simplices.copy())
        area = tmesh.entity_measure('cell')
        tmesh.delete_cell(area < 1e-8)
        area = tmesh.entity_measure('cell')

       

show_error_table(Ndof, errorType, errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
