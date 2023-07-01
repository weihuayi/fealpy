
import numpy as np
import sys

from fealpy.model.obstacle_model_2d import ObstacleData1, ObstacleData2 
from fealpy.vemmodel.ObstacleVEMModel2d import ObstacleVEMModel2d
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.quadrature import TriangleQuadrature 
from fealpy.mesh import PolygonMesh
from fealpy.mesh.simple_mesh_generator import distmesh2d, drectangle
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh

import matplotlib.pyplot as plt
import scipy.io as sio

def load_mesh(f):
    data = sio.loadmat('../meshdata/'+f)
    point = data['point']
    cell = np.array(data['cell'].reshape(-1), dtype=np.int)
    cellLocation = np.array(data['cellLocation'].reshape(-1), dtype=np.int)
    mesh = PolygonMesh(point, cell, cellLocation)
    return mesh


fd = lambda p: drectangle(p, [-2, 2, -2, 2])
bbox = [-2.2, 2.2, -2.2, 2.2]
pfix = np.array([[-2, -2],[2, -2], [2, 2],[-2, 2]], dtype=np.float) 

m = int(sys.argv[1])
maxit = int(sys.argv[2])
p = int(sys.argv[3])

if m == 1:
    model = ObstacleData1()
    quadtree= model.init_mesh(n=3, meshtype='quadtree')
elif m == 2:
    model = ObstacleData2() 
    #mesh = load_mesh('nonconvexpmesh1.mat')
    #h0 = 0.2
    #mesh = distmesh2d(fd, h0, bbox, pfix, meshtype='polygon')
    n = 20
    mesh = rectangledomainmesh([-2, 2, -2, 2], nx=n, ny=n, meshtype='polygon')

errorType = ['$\| u - \Pi^\\nabla u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\\nabla u_h\|$'
             ]

integrator = TriangleQuadrature(3)
vem = ObstacleVEMModel2d(model, mesh, p=p, integrator=integrator)

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

data = {}
for i in range(maxit):
    print('step:', i)
    vem.solve(solver='direct')
    Ndof[i] = vem.vemspace.number_of_global_dofs()
    errorMatrix[0, i] = vem.L2_error()
    errorMatrix[1, i] = vem.H1_semi_error()

    data['uh{}'.format(i+1)] = vem.uh
    data['elem{}'.format(i+1)] = mesh.ds.cell
    data['elemLocation{}'.format(i+1)] = mesh.ds.cellLocation
    data['node{}'.format(i+1)] = mesh.point
    if i < maxit - 1:
        #fi = 'nonconvexpmesh{}.mat'
        #mesh = load_mesh(fi.format(i+2))
        #h0 /= 2
        #mesh = distmesh2d(fd, h0, bbox, pfix, meshtype='polygon')
        n *= 2
        mesh = rectangledomainmesh([-2, 2, -2, 2], nx=n, ny=n, meshtype='polygon')
        vem.reinit(mesh)

data['errorMatrix'] = errorMatrix
data['Ndof'] = Ndof 
fo = 'uh{}.mat'.format(p)
sio.matlab.savemat(fo.format(i+1), data)

mesh.add_plot(plt, cellcolor='w')
show_error_table(Ndof, errorType, errorMatrix)
#fig2 = plt.figure()
#fig2.set_facecolor('white')
#axes = fig2.gca(projection='3d')
#x = mesh.point[:, 0]
#y = mesh.point[:, 1]
#tri = quadtree.leaf_cell(celltype='tri')
#s = axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
#fig2.colorbar(s)
#
#fig3 = plt.figure()
#fig3.set_facecolor('white')
#axes = fig3.gca(projection='3d')
#s = axes.plot_trisurf(x, y, tri, vem.uI-vem.gI, cmap=plt.cm.jet, lw=0.0)
#fig3.colorbar(s)

showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
