
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os
import scipy.io as sio


from fealpy.fem.BiharmonicFEMModel import BiharmonicRecoveryFEMModel
from fealpy.pde.BiharmonicModel2d import BiharmonicData4, BiharmonicData5, BiharmonicData6, BiharmonicData7, BiharmonicData8

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
    pde = BiharmonicData5(a=0.01)
    mesh = pde.init_mesh()
elif m == 2:
    pde = BiharmonicData6()
    mesh = pde.init_mesh(n=4)
elif m == 3:
    pde = BiharmonicData7()
    mesh = pde.init_mesh(n=1)
elif m == 4:
    pde = BiharmonicData8()
    mesh = pde.init_mesh(n=1)
else:
    raise ValueError("error!")


sigma = 1
k = maxit -20  
degree = 1


idx = [0] + list(range(9, maxit, 10))

errorType = [
         '$\| u - u_h\|$',
         '$\|\\nabla u - \\nabla u_h\|$',
         '$\|\\nabla u_h - G(\\nabla u_h) \|$',
         '$\|\\nabla u - G(\\nabla u_h)\|$',
         '$\|\Delta u - \\nabla\cdot G(\\nabla u_h)\|$',
         '$\|\Delta u -  G(\\nabla\cdot G(\\nabla u_h))\|$',
         '$\|G(\\nabla\cdot G(\\nabla u_h)) - \\nabla\cdot G(\\nabla u_h)\|$'
         ]

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

dirichlet = True
for i in range(maxit):
    print(i, 'step:')
    fem = BiharmonicRecoveryFEMModel(mesh, pde, 1, 3, rtype='harmonic', dirichlet=dirichlet)
    fem.solve()
    eta0 = fem.grad_recover_estimate()
    eta1 = fem.laplace_recover_estimate(etype=1)

    Ndof[i] = len(fem.uh) 

    e0, e1, e2, e3, e4 = fem.get_error()
    eta0 = fem.grad_recover_estimate()
    eta1 = fem.laplace_recover_estimate(etype=1)
    e5 = np.sqrt(np.sum(eta0**2))
    e6 = np.sqrt(np.sum(eta1**2))

    errorMatrix[0, i] = e0 # L2
    errorMatrix[1, i] = e1 # H1
    errorMatrix[2, i] = e5 # grad u_h - G(grad u_h)
    errorMatrix[3, i] = e2 # grad u - G(grad u_h)
    errorMatrix[4, i] = e3 # Delta u - div G(grad u_h)
    errorMatrix[5, i] = e4 # Delta u - G(div G(grad u_h))
    errorMatrix[6, i] = e6 # div G(grad u_h) - G(div G(grad u_h))

    if i in idx:
        fig = plt.figure()
        fig.set_facecolor('white')
        axes = fig.gca() 
        mesh.add_plot(axes, cellcolor='w')
        fig.savefig(d+'/mesh'+str(m-2)+'-'+str(i)+'.pdf')


    markedCell = mark(eta1, theta)
    if i < maxit - 1:
        mesh.bisect(markedCell)

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
node = mesh.entity('node')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
s = axes.plot_trisurf(x, y, fem.uh, triangles=mesh.ds.cell, cmap=plt.cm.jet, lw=0.0)
fig2.colorbar(s)
fig2.savefig(d+'/solution.pdf')

fig3 = plt.figure(figsize=(40, 40), facecolor='w')
axes = fig3.gca()
axes = showmultirate(axes, k, Ndof, errorMatrix[4:7:2, :],  errorType[4:7:2])

#optionlist = ['k-*', 'r-^']
#showmultirate(axes, k, Ndof, errorMatrix[4:7:2], optionlist, errorType[4:7:2])
axes.legend(loc=3, prop={'size': 60})
axes.tick_params(labelsize=60)
axes.axis('tight')
fig3.savefig(d+'/error.pdf')
plt.show()
