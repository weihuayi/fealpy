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
        isLeftCell = np.zeros((n, n), dtype=np.bool)
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
        return TriangleMesh(node, newCell)

    def rice_mesh(self, box, n=10):
        qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
        node = qmesh.entity('node')
        cell = qmesh.entity('cell')
        NC = qmesh.number_of_cells()

        isLeftCell = np.zeros((n, n), dtype=np.bool)
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

    def random_mesh(self, box, n=10):
        h = (box[1] - box[0])/n
        return triangle(box, h, meshtype='tri')

        
m = int(sys.argv[1])
n = int(sys.argv[2])
meshtype = int(sys.argv[3])
maxit = 4

if m == 1:
    model = CosCosData()
elif m == 2:
    model = ExpData()
elif m == 3:
    model = SinSinData()
elif m == 4:
    model = PolynomialData

if meshtype == 0:
    mesh = model.init_mesh(n=n, meshtype='tri')
elif meshtype == 1:
    mesh = Meshtype.fishbone(n, meshtype="uniform")
elif meshtype == 1:
    mesh = Meshtype.cross_mesh(n)
elif meshtype == 2:
    mesh = Meshtype.rice_mesh(n)

h0 = 0.5

ralg = FEMFunctionRecoveryAlg()


errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h||_{0}$',
             '$||\\nabla u - \\nabla u_h||_{0}$', 
             '$||\\nabla u - G(\\nabla u_h)||_{0}simple$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}area$',
             '$||\\nabla u - G(\\nabla u_h)||_{0}har$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

for i in range(maxit):
    if meshtype == 3:
        mesh = Meshtype.random_mesh(h0/2**i)
    elif meshtype == 4:
        mesh = load_mat_mesh('../data/ll/chevronmesh'+str(i+1)+'.mat')
    elif meshtype == 5:
        mesh = load_mat_mesh('../data/ll/crisscrossmesh'+str(i+1)+'.mat')       
    elif meshtype == 6:
        mesh = load_mat_mesh('../data/ll/gtrimesh'+str(i+1)+'.mat')
    elif meshtype == 7:
        mesh = load_mat_mesh('../data/ll/unionjackmesh'+str(i+1)+'.mat')

    mesh.add_plot(plt)
    fem = PoissonFEMModel(mesh, model, 1)
    fem.solve()
    uh = fem.uh
    Ndof[i] = fem.mesh.number_of_cells() 
    errorMatrix[0, i] = fem.l2_error()
    errorMatrix[1, i] = fem.L2_error()
    errorMatrix[2, i] = fem.H1_semi_error()
    rguh = ralg.simple_average(uh)
    errorMatrix[3, i] = fem.recover_error(rguh)
    rguh1 = ralg.area_average(uh)
    errorMatrix[4, i] = fem.recover_error(rguh1)
    rguh2 = ralg.harmonic_average(uh)
    errorMatrix[5, i] = fem.recover_error(rguh2)

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
