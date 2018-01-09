
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os
import scipy.io as sio


from fealpy.femmodel.BiharmonicFEMModel import BiharmonicRecoveryFEMModel
from fealpy.model.BiharmonicModel2d import BiharmonicData4, BiharmonicData5, BiharmonicData6, BiharmonicData7, BiharmonicData8

from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 

from fealpy.tools.show import showmultirate

from fealpy.mesh.adaptive_tools import mark


def smooth_eta(mesh, eta):
    m = np.max(eta)
    eta /= m
    area = mesh.area()
    q2c = mesh.ds.point_to_cell()
    cell = mesh.ds.cell
    w = q2c@area
    for i in range(10): 
        beta = q2c@(eta*area)/w
        eta = np.sum(beta[cell], axis=1)/3 
    return eta
        


m = int(sys.argv[1])
theta = float(sys.argv[2])
maxit = int(sys.argv[3])
d = sys.argv[4]

if not os.path.exists(d):
    os.mkdir(d)

if m == 1:
    model = BiharmonicData5(a=0.01)
    mesh = model.init_mesh()
elif m == 2:
    model = BiharmonicData6()
    mesh = model.init_mesh(n=4)
elif m == 3:
    model = BiharmonicData7()
    mesh = model.init_mesh(n=1)
elif m == 4:
    model = BiharmonicData8()
    mesh = model.init_mesh(n=1)
else:
    raise ValueError("error!")


sigma = 1
k = maxit -20  
degree = 1


idx = [0] + list(range(9, maxit, 10))

errorType = ['$\| u - u_h\|$',
         '$\|\\nabla u - \\nabla u_h\|$',
         '$\|\\nabla u_h - G(\\nabla u_h) \|$',
         '$\|\\nabla u - G(\\nabla u_h)\|$',
         '$\|\Delta u - \\nabla\cdot G(\\nabla u_h)\|$',
         '$\|\Delta u -  G(\\nabla\cdot G(\\nabla u_h))\|$',
         '$\|G(\\nabla\cdot G(\\nabla u_h))-\\nabla\cdot G(\\nabla u_h)\|$',
         '$\|\Delta u\|$',
         '$\|\\nabla\cdot G(\\nabla u_h)\|$',
         '$\|G(\\nabla\cdot G(\\nabla u_h))\|$'
         ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

integrator = TriangleQuadrature(3)

for i in range(maxit):
    print(i, 'step:')

    fem = BiharmonicRecoveryFEMModel(mesh, model, integrator, rtype='harmonic')
    fem.solve()
    fem.recover_grad()
    fem.recover_laplace()

    eta1 = fem.grad_recover_estimate()

    eta2 = fem.laplace_recover_estimate(etype=1)
    eta3 = fem.laplace_recover_estimate(etype=2)
    eta4 = fem.laplace_recover_estimate(etype=3)

    Ndof[i] = len(fem.uh) 
    errorMatrix[0, i] = fem.funNorm.L2_error(model.solution, fem.uh)
    errorMatrix[1, i] = fem.H1_semi_error(model.gradient, fem.uh)
    errorMatrix[2, i] = np.sqrt(np.sum(eta1**2))
    errorMatrix[3, i] = fem.funNorm.L2_error(model.gradient, fem.rgh)
    errorMatrix[4, i] = fem.funNorm.div_error(model.laplace, fem.rgh)
    errorMatrix[5, i] = fem.funNorm.L2_error(model.laplace, fem.rlh)
    errorMatrix[6, i] = np.sqrt(np.sum(eta2**2))
    errorMatrix[7, i] = fem.funNorm.L2_norm(model.laplace, fem.V)
    errorMatrix[8, i] = np.sqrt(np.sum(eta3**2))
    errorMatrix[9, i] = np.sqrt(np.sum(eta4**2))

    if i in idx:
        fig = plt.figure()
        fig.set_facecolor('white')
        axes = fig.gca() 
        mesh.add_plot(axes, cellcolor='w')
        fig.savefig(d+'/mesh'+str(m-2)+'-'+str(i)+'.pdf')


    markedCell = mark(eta2, theta)
    if i < maxit - 1:
        mesh.bisect(markedCell)

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
s = axes.plot_trisurf(x, y, uh, triangles=mesh.ds.cell, cmap=plt.cm.jet, lw=0.0)
fig2.colorbar(s)
fig2.savefig(d+'/solution.pdf')

fig3 = plt.figure(figsize=(40, 40), facecolor='w')
axes = fig3.gca()

optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, k, Ndof, errorMatrix, optionlist, errorType)

#optionlist = ['k-*', 'r-^']
#showmultirate(axes, k, Ndof, errorMatrix[4:7:2], optionlist, errorType[4:7:2])
axes.legend(loc=3, prop={'size': 60})
axes.tick_params(labelsize=60)
axes.axis('tight')
fig3.savefig(d+'/error.pdf')
plt.show()
