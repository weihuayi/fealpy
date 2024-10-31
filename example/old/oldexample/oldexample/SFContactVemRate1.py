import numpy as np
import sys

from fealpy.model.simplified_frictional_contact_problem import SFContactProblemData, SFContactProblemData1
from fealpy.vemmodel.SFContactVEMModel2d import SFContactVEMModel2d 
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.quadrature import TriangleQuadrature 
from fealpy.mesh import PolygonMesh
from fealpy.mesh.simple_mesh_generator import distmesh2d, drectangle, triangle
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh
import matplotlib.pyplot as plt
import scipy.io as sio


fd = lambda p: drectangle(p, [0, 1, 0, 1])
bbox = [-0.2, 1.2, -0.2, 1.2]
pfix = np.array([[0, 0],[1, 0], [1, 1],[0, 1]], dtype=np.float) 

m = int(sys.argv[1])
maxit = int(sys.argv[2])
mtype = int(sys.argv[3])

def load_mesh(f):
    data = sio.loadmat('../meshdata/'+f)
    point = data['point']
    cell = np.array(data['cell'].reshape(-1), dtype=np.int)
    cellLocation = np.array(data['cellLocation'].reshape(-1), dtype=np.int)
    mesh = PolygonMesh(point, cell, cellLocation)
    return mesh

if m == 1:
    model = SFContactProblemData()

integrator = TriangleQuadrature(4)

if mtype == 1:
    pmesh = load_mesh('nonconvexpmesh1.mat')
    pmesh.point += 2.0
    pmesh.point /= 4.0
else:
    h0 = 0.2
    #pmesh = distmesh2d(fd, h0, bbox, pfix, meshtype='polygon')
    pmesh = triangle([0, 1, 0, 1], h0, meshtype='polygon')
    #n = 4 
    #pmesh = rectangledomainmesh([0, 1, 0, 1], nx=n, ny=n, meshtype='polygon')

Ndof = np.zeros((maxit,), dtype=np.int)
vem = SFContactVEMModel2d(model, pmesh, p=1, integrator=integrator)
solution = {}
for i in range(maxit):
    print('step:', i)
    vem.solve(rho=0.1, maxit=40000)
    Ndof[i] = vem.vemspace.number_of_global_dofs()

    NC = pmesh.number_of_cells()
    cell = np.zeros(NC, dtype=np.object)
    cell[:] = np.vsplit(pmesh.ds.cell.reshape(-1, 1)+1, pmesh.ds.cellLocation[1:-1])


    solution['mesh{}'.format(i)] = {
            'vertices':pmesh.point,
            'elements':cell.reshape(-1, 1),
            'boundary':pmesh.ds.boundary_edge()+1,
            'solution':vem.uh.reshape(-1, 1)}

    if i < maxit - 1:
        if mtype == 1:
            fi = 'nonconvexpmesh{}.mat'
            pmesh = load_mesh(fi.format(i+2))
            pmesh.point += 2.0
            pmesh.point /= 4.0
        else:
            h0 /= 2
            #pmesh = distmesh2d(fd, h0, bbox, pfix, meshtype='polygon')
            pmesh = triangle([0, 1, 0, 1], h0, meshtype='polygon')
            #n *= 2 
            #pmesh = rectangledomainmesh([0, 1, 0, 1], nx=n, ny=n, meshtype='polygon')
        vem.reinit(pmesh)

solution['Ndof'] = Ndof
f = 'solution.mat'
sio.matlab.savemat(f, solution)
