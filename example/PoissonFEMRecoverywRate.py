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
    def __init__(self):
        pass

    #Fishbone
    def fishbone(self, n=4, meshtype="uniform_bisect"):
        node = np.array([
            (0, 0),
            (1/2, 0),
            (1, 0),
            (0, 1/2),
            (1/2, 1/2),
            (1, 1/2),
            (0, 1),
            (1/2, 1),
            (1, 1)], dtype=np.float)
        cell=np.array([
            (3, 0, 4), 
            (1, 4, 0), 
            (1, 2, 4),
            (5, 4, 2),
            (6, 3, 7),
            (4, 7, 3),
            (4, 5, 7),
            (8, 7, 5)], dtype=np.int)
        mesh = TriangleMesh(node, cell)        
        mesh.uniform_bisect(n)       
        return mesh

    #cross mesh
    def cross_mesh(self, n=4, meshtype="uniform_bisect"):
        node=np.array([
            (0, 0),
            (1, 0),
            (1/2, 1/2),
            (0, 1),
            (1, 1)], detype=np.float)
        cell=np.array([
            (2, 0, 1),
            (2, 3, 0),
            (2, 1, 0),
            (2, 4, 3)], dtype=np.int)
        mesh = TriangleMesh(node, cell)        
        mesh.uniform_bisect(n)      
        return mesh

    def rice_mesh(self, n=4, meshtype="uniform_bisect"):
        node = np.array([
            (0, 0),
            (1/2, 0),
            (1, 0),
            (0, 1/2),
            (1/2, 1/2),
            (1, 1/2),
            (0, 1),
            (1/2, 1),
            (1, 1)], dtype=np.float)

        cell = np.array([
            (1, 4, 0),
            (3, 0, 4),
            (1, 2, 4),
            (5, 4, 2),
            (3, 4, 6),
            (7, 6, 4),
            (5, 8, 4),            
            (7, 4, 8)], dtype=np.int)
        mesh = TriangleMesh(node, cell)        
        mesh.uniform_bisect(n)       
        return mesh

    def random_mesh(self, h=0.5):

        mesh_info = MeshInfo()
        mesh_info.set_points([(0,0), (1,0), (1,1), (0,1)])
        mesh_info.set_facets([[0,1], [1,2], [2,3], [3,0]]) 

        mesh = build(mesh_info, max_volume=(h)**2)
        point = np.array(mesh.points, dtype=np.float)
        cell = np.array(mesh.elements, dtype=np.int)
        mesh = TriangleMesh(point, cell)      
        return mesh
        
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

    if i < maxit - 1:
        mesh.uniform_refine()
        fem.reinit(mesh)

print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
