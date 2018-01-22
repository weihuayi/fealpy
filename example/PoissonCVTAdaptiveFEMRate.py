import sys

import numpy as np  
import matplotlib.pyplot as plt
from fealpy.model.poisson_model_2d import LShapeRSinData
from fealpy.femmodel.PoissonFEMModel import PoissonFEMModel

from fealpy.tools.show import showmultirate

from fealpy.functionspace import FunctionNorm
from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 

from fealpy.mesh import meshio 

def node_sizing(mesh, e, eta=0.8, smooth=8):
    area = mesh.area()
    h = np.sqrt(4*area/np.sqrt(3))
    N = mesh.number_of_cells()
    ep = eta*np.min(e)
    print(np.max(h))
    ch = np.mean(h)*(ep/e)**2
    p2c = mesh.ds.point_to_cell()
    valence = np.asarray(p2c.sum(axis=1)).reshape(-1)
    cell = mesh.ds.cell
    for i in range(smooth):
        ph = np.asarray(p2c@ch).reshape(-1)/valence 
        ch = np.sum(ph[cell], axis=1)/3

    return ph

m = int(sys.argv[1])
n = int(sys.argv[2])

if m == 1:
    model = LShapeRSinData() 

mesh = model.init_mesh(n=n, meshtype='tri')
integrator = TriangleQuadrature(3)
fem = PoissonFEMModel(mesh, model, p=1, integrator=integrator)
funNorm = FunctionNorm(integrator, fem.area)
maxit = 2

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$',
             '$||G(\\nabla u_h) - \\nabla u_h||_{0}$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
ne = len(errorType)
errorMatrix = np.zeros((ne, maxit), dtype=np.float)
for i in range(maxit):
    fem.solve()
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = funNorm.l2_error(model.solution, fem.uh)
    errorMatrix[1, i] = funNorm.L2_error(model.solution, fem.uh)
    errorMatrix[2, i] = funNorm.H1_semi_error(model.gradient, fem.uh)
    e = fem.recover_estimate()
    h = node_sizing(mesh, e, eta=1, smooth=6)
    errorMatrix[3, i] = np.sqrt(np.sum(e**2))
    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)
        funNorm.area = fem.area
        
meshio.write_obj_mesh(mesh, 'squaretrimesh.obj')
np.savetxt('size.txt', h)

fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca() 
mesh.add_plot(axes, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
s = axes.plot_trisurf(x, y, h, triangles=mesh.ds.cell, cmap=plt.cm.jet, lw=0.0)
fig2.colorbar(s)

plt.show()

#print('Ndof:', Ndof)
#print('error:', errorMatrix)
#fig = plt.figure()
#fig.set_facecolor('white')
#axes = fig.gca()
#optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
#showmultirate(axes, 0, Ndof, errorMatrix[:ne, :], optionlist[:ne], errorType[:ne])
#axes.legend(loc=3, prop={'size': 30})
#plt.show()
