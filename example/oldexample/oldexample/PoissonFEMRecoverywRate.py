import sys
import numpy as np  
import matplotlib.pyplot as plt

from fealpy.model.poisson_model_2d import CosCosData, SinSinData, ExpData, PolynomialData

from fealpy.femmodel.PoissonFEMModel import PoissonFEMModel
from fealpy.tools.show import showmultirate
from fealpy.recovery import FEMFunctionRecoveryAlg

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.meshio import load_mat_mesh, write_mat_mesh, write_mat_linear_system 
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh  
from fealpy.mesh.simple_mesh_generator import triangle
from meshpy.triangle import MeshInfo, build


class Meshtype():
    #Fishbone
    def regular(self, box, n=10):
        return rectangledomainmesh(box, nx=n, ny=n, meshtype='tri')

    def fishbone(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NC = qmesh.number_of_cells()
        isLeftCell = np.zeros((n, n), dtype=np.bool_)
        isLeftCell[0::2, :] = True
        isLeftCell = isLeftCell.reshape(-1)
        lcell = cell[isLeftCell]
        rcell = cell[~isLeftCell]
        newCell = np.r_['0', 
                lcell[:, [1, 2, 0]], 
                lcell[:, [3, 0, 2]],
                rcell[:, [0, 1, 3]],
                rcell[:, [2, 3, 1]]]
        return TriangleMesh(node, newCell)

        #cross mesh
    def cross_mesh(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NN = qmesh.number_of_nodes()
        NC = qmesh.number_of_cells()
        bc = qmesh.barycenter('cell') 
        newNode = np.r_['0', node, bc]

        newCell = np.zeros((4*NC, 3), dtype=np.int) 
        newCell[0:NC, 0] = range(NN, NN+NC)
        newCell[0:NC, 1:3] = cell[:, 0:2]
        
        newCell[NC:2*NC, 0] = range(NN, NN+NC)
        newCell[NC:2*NC, 1:3] = cell[:, 1:3]

        newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
        newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

        newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
        newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
        return TriangleMesh(newNode, newCell)

    def rice_mesh(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NC = qmesh.number_of_cells()

        isLeftCell = np.zeros((n, n), dtype=np.bool_)
        isLeftCell[0, 0::2] = True
        isLeftCell[1, 1::2] = True
        if n > 2:
            isLeftCell[2::2, :] = isLeftCell[0, :]
        if n > 3:
            isLeftCell[3::2, :] = isLeftCell[1, :]
        isLeftCell = isLeftCell.reshape(-1)

        lcell = cell[isLeftCell]
        rcell = cell[~isLeftCell]
        newCell = np.r_['0', 
                lcell[:, [1, 2, 0]], 
                lcell[:, [3, 0, 2]],
                rcell[:, [0, 1, 3]],
                rcell[:, [2, 3, 1]]]
        return TriangleMesh(node, newCell)

    def delaunay_mesh(self, box, n=10):
        h = (box[1] - box[0])/n
        return triangle(box, h, meshtype='tri')

    def nonuniform_mesh(self, box, n=10):
        nx = 4*n
        ny = 4*n
        n1 = 2*n+1

        N = n1**2
        NC = 4*n*n
        node = np.zeros((N,2))
        
        x = np.zeros(n1, dtype=np.float)
        x[0::2] = range(0,nx+1,4)
        x[1::2] = range(3, nx+1, 4)

        y = np.zeros(n1, dtype=np.float)
        y[0::2] = range(0,nx+1,4)
        y[1::2] = range(1, nx+1,4)

        node[:,0] = x.repeat(n1)/nx
        node[:,1] = np.tile(y, n1)/ny


        idx = np.arange(N).reshape(n1, n1)
        
        cell = np.zeros((2*NC, 3), dtype=np.int)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
        return TriangleMesh(node, cell)
        
    def uncross_mesh(self, box, n=10, r="1"):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NN = qmesh.number_of_nodes()
        NC = qmesh.number_of_cells()   
        bc = qmesh.barycenter('cell') 

        if r=="1":
            bc1 = np.sqrt(np.sum((bc-node[cell[:,0], :])**2, axis=1))[0]
            newNode = np.r_['0', node, bc-bc1*0.3]
        elif r=="2":
            ll = node[cell[:, 0]] - node[cell[:, 2]]
            bc = qmesh.barycenter('cell') + ll/4
            newNode = np.r_['0',node, bc]

        newCell = np.zeros((4*NC, 3), dtype=np.int) 
        newCell[0:NC, 0] = range(NN, NN+NC)
        newCell[0:NC, 1:3] = cell[:, 0:2]
            
        newCell[NC:2*NC, 0] = range(NN, NN+NC)
        newCell[NC:2*NC, 1:3] = cell[:, 1:3]

        newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
        newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

        newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
        newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
        return TriangleMesh(newNode, newCell)

        
m = int(sys.argv[1])
meshtype = int(sys.argv[2])
n = int(sys.argv[3])
maxit = 4
box=[0,1,0,1]
if m == 1:
    pde = CosCosData()
elif m == 2:
    pde = ExpData()
elif m == 3:
    pde = PolynomialData()
elif m == 4:
    pde = SinSinData()

Mesh = Meshtype()


ralg = FEMFunctionRecoveryAlg()

errorType = [#'$|| u_I - u_h ||_{l_2}$',
             #'$|| u - u_h||_{0}$',
             #'$||\\nabla u - \\nabla u_h||_{0}$', 
             '$||\\nabla u - G(\\nabla u_h)||_{0}sim$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}are$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}har$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}angle$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}dhar$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}scr$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}zz$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}ppr$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)


for i in range(maxit):
    if meshtype == 1:
        mesh = Mesh.regular(box, n=n)
    elif meshtype == 2:
        mesh = Mesh.cross_mesh(box, n=n)
    elif meshtype == 3:
        mesh = Mesh.rice_mesh(box, n=n)
    elif meshtype == 4:
        mesh = Mesh.fishbone(box, n=n)
    elif meshtype == 5:
        mesh = Mesh.delaunay_mesh(box, n=n)
    elif meshtype == 6:
        mesh = Mesh.uncross_mesh(box, n=n, r='1')
    elif meshtype == 7:
        mesh = Mesh.nonuniform_mesh(box, n=n)

    integrator = mesh.integrator(3)
    mesh.add_plot(plt,cellcolor='w')
    fem = PoissonFEMModel(pde, mesh, 1, integrator)
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_cells() 
    #errorMatrix[0, i] = fem.l2_error()
    #errorMatrix[1, i] = fem.L2_error()
    #errorMatrix[2, i] = fem.H1_semi_error()
    rguh = ralg.simple_average(uh)
    errorMatrix[0, i] = fem.recover_error(rguh)
    rguh1 = ralg.area_average(uh)
    errorMatrix[1, i] = fem.recover_error(rguh1)
    rguh2 = ralg.distance_harmonic_average(uh)
    errorMatrix[2, i] = fem.recover_error(rguh2)
    rguh3 = ralg.angle_average(uh)
    errorMatrix[3, i] = fem.recover_error(rguh3)
    rguh4 = ralg.distance_harmonic_average(uh)
    errorMatrix[4, i] = fem.recover_error(rguh4)
    rguh5 = ralg.SCR(uh)
    errorMatrix[5, i] = fem.recover_error(rguh5)
    rguh6 = ralg.ZZ(uh)
    errorMatrix[6, i] = fem.recover_error(rguh6)
    rguh7 = ralg.PPR(uh)
    errorMatrix[7, i] = fem.recover_error(rguh7)
    if i < maxit - 1:
        n *= 2

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 2, Ndof, errorMatrix, errorType)
plt.show()
