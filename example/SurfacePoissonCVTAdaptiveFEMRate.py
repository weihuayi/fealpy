
import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.model.surface_poisson_model_3d import SphereSinSinSinData
from fealpy.femmodel.SurfacePoissonFEMModel import SurfacePoissonFEMModel
from fealpy.quadrature import TriangleQuadrature 

from fealpy.mesh import meshio 
from fealpy.tools.show import showmultirate


def node_sizing(mesh, e, eta=0.8, smooth=8):
    area = mesh.area()
    h = np.sqrt(4*area/np.sqrt(3))
    N = mesh.number_of_cells()
    ep = eta*np.min(e)
    print(np.mean(h))
    ch = np.mean(h)*(ep/e)**2
    p2c = mesh.ds.point_to_cell()
    valence = np.asarray(p2c.sum(axis=1)).reshape(-1)
    cell = mesh.ds.cell
    for i in range(smooth):
        ph = np.asarray(p2c@ch).reshape(-1)/valence 
        ch = np.sum(ph[cell], axis=1)/3

    return ph, ch

m = int(sys.argv[1])
p = int(sys.argv[2]) 
q = int(sys.argv[3])


if m == 1:
    model = SphereSinSinSinData()
    surface = model.surface 
    mesh = surface.init_mesh()
    mesh.uniform_refine(n=2, surface=surface)

integrator = TriangleQuadrature(q)
fem = SurfacePoissonFEMModel(mesh, surface, model, p=p, integrator=integrator)
maxit = 2

errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{S,0}$',
             '$||\\nabla_S u - \\nabla_S u_h||_{S, 0}$',
             '$||G(\\nabla_S u_h) - \\nabla_S u_h||_{S_h, 0}$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit):
    fem.solve()
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    e = fem.recover_estimate()
    bc = mesh.barycenter()
    e = 1/np.exp(bc[:, 2]**2)**2
    h , ch= node_sizing(mesh, e, eta=1, smooth=6)
    errorMatrix[3, i] = np.sqrt(np.sum(e**2))
    if i < maxit - 1:
        mesh.uniform_refine(1, surface)
        fem.reinit(mesh)

meshio.write_obj_mesh(mesh, 'spheretrimesh.obj')
np.savetxt('size.txt', h)

fig = plt.figure()
fig.set_facecolor('white')
axes = fig.gca(projection='3d')
x = mesh.point[:, 0]
y = mesh.point[:, 1]
z = mesh.point[:, 2] 
mesh.add_plot(axes, cellcolor=ch, showcolorbar=True)
#axes.plot_trisurf(x, y, z, triangles=mesh.ds.cell, cmap=plt.cm.jet, lw=0.0)

fig = plt.figure()
fig.set_facecolor('white')
axes = fig.gca()
optionlist = ['k-*', 'b-o', 'r--^', 'g->', 'm-8', 'c-D','y-x', 'y-+', 'y-h', 'y-p']
showmultirate(axes, 0, Ndof, errorMatrix[:4, :], optionlist[:4], errorType[:4])
axes.legend(loc=3, prop={'size': 30})
plt.show()
