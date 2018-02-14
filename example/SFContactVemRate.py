
import numpy as np
import sys

from fealpy.model.simplified_frictional_contact_problem import SFContactProblemData 
from fealpy.vemmodel.SFContactVEMModel2d import SFContactVEMModel2d 
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.quadrature import TriangleQuadrature 

import matplotlib.pyplot as plt
import scipy.io as sio

maxit = int(sys.argv[1])

model = SFContactProblemData()
quadtree= model.init_mesh(n=3, meshtype='quadtree')

integrator = TriangleQuadrature(4)

pmesh = quadtree.to_pmesh()
vem1 = SFContactVEMModel2d(model, pmesh, p=1, integrator=integrator)
vem2 = SFContactVEMModel2d(model, pmesh, p=2, integrator=integrator)

Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$\| u - \Pi^\\nabla u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\\nabla u_h\|$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    print('step:', i)
    vem1.solve(rho=10)
    Ndof[i] = vem1.vemspace.number_of_global_dofs()
    vem2.solve(rho=10)
    errorMatrix[0, i] = vem2.L2_error(u=vem1.S.value)
    errorMatrix[1, i] = vem2.H1_semi_error(gu=vem1.S.grad_value)
    if i < maxit - 1:
        quadtree.uniform_refine()
        pmesh = quadtree.to_pmesh()
        vem1.reinit(pmesh)
        vem2.reinit(pmesh)

show_error_table(Ndof, errorType, errorMatrix)

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = pmesh.point[:, 0]
y = pmesh.point[:, 1]
tri = quadtree.leaf_cell(celltype='tri')
s = axes.plot_trisurf(x, y, tri, vem2.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
fig2.colorbar(s)
plt.show()

#    p = quadtree.point.copy()
#    cell = quadtree.ds.cell
#    isLeafCell = quadtree.is_leaf_cell()
#    cell = cell[isLeafCell].copy()
#    solution.append([vem.uh.copy(), p, cell])
