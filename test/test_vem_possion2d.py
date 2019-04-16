import numpy as np
import scipy.io as sio

from fealpy.pde.poisson_model_2d import CrackData, LShapeRSinData, CosCosData, KelloggData, SinSinData
from fealpy.vem import PoissonVEMModel 
from fealpy.tools.show import showmultirate
from fealpy.mesh.simple_mesh_generator import triangle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityNode
from fealpy.mesh.PolygonMesh import PolygonMesh
from fealpy.mesh.simple_mesh_generator import distmesh2d 
from fealpy.mesh.level_set_function import drectangle
import triangle as tri
from scipy.spatial import Delaunay


pde = CosCosData()
maxit = 5
h = 0.2
box = [0, 1, 0, 1]
pfix = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]], dtype=np.float)
fd = lambda p: drectangle(p, box)

pmesh = distmesh2d(fd, h, box, pfix, meshtype='polygon')
area = pmesh.entity_measure('cell')

node = pmesh.entity('node')
t = Delaunay(node)
tmesh = TriangleMesh(node, t.simplices.copy())

area = tmesh.entity_measure('cell')
tmesh.delete_cell(area < 1e-8)
area = tmesh.entity_measure('cell')
print(len(tmesh.node))
pmesh = PolygonMesh.from_mesh(tmesh)
print(pmesh.ds.cell)
fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
#mesh.find_node(axes, showindex=True)
plt.show()

Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u - \Pi^\\nabla u_h||_0$ with p=1',
             '$||\\nabla u - \\nabla \Pi^\\nabla u_h||_0$ with p=1',
             ]

p = 1

errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
integrator = pmesh.integrator(7)


for i in range(maxit):
    vem = PoissonVEMModel(pde, pmesh, p, integrator)
    ls = vem.solve()
    Ndof[i] = vem.vemspace.number_of_global_dofs()
    errorMatrix[0, i] = vem.L2_error()
    errorMatrix[1, i] = vem.H1_semi_error()
    if i < maxit - 1:
        #tmesh.uniform_refine()
        #pmesh = PolygonMesh.from_mesh(tmesh)
        h = h/2
        pmesh = distmesh2d(fd, h, box, pfix, meshtype='polygon')
        area = pmesh.entity_measure('cell')
        node = pmesh.entity('node')
        t = Delaunay(node)
        tmesh = TriangleMesh(node, t.simplices.copy())
        area = tmesh.entity_measure('cell')
        tmesh.delete_cell(area < 1e-8)
        pmesh = PolygonMesh.from_mesh(tmesh)

print(errorMatrix)
print(Ndof)
print(len(pmesh.node))
pmesh.add_plot(plt, cellcolor='w')
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
