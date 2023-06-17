#!/usr/bin/env python3
#
import numpy as np

from fealpy.pde.sfc_2d import SFCModelData
from fealpy.vem.SFCVEMModel2d import SFCVEMModel2d
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.quadrature import TriangleQuadrature
from fealpy.mesh import PolygonMesh

import matplotlib.pyplot as plt
import scipy.io as sio

maxit = 6

model = SFCModelData()
qmesh = model.init_mesh(n=2, meshtype='quad')

integrator = TriangleQuadrature(4)

pmesh = PolygonMesh.from_mesh(qmesh)
vem = SFCVEMModel2d(model, pmesh, p=1, integrator=integrator)

Ndof = np.zeros((maxit,), dtype=np.int)
data = []
solution = {}
for i in range(maxit):
    print('step:', i)
    vem.solve(rho=0.1, maxit=40000)
    Ndof[i] = vem.vemspace.number_of_global_dofs()

    NC = qmesh.number_of_cells()
    cell = np.zeros(NC, dtype=np.object)
    cell[:] = list(qmesh.ds.cell+1)

    solution['mesh{}'.format(2+i)] = {
            'vertices': qmesh.point,
            'elements': cell.reshape(-1, 1),
            'boundary': qmesh.ds.boundary_edge()+1,
            'solution': vem.uh.reshape(-1, 1)}
    if i < maxit - 1:
        data.append(vem.uh.copy())
        edge = qmesh.ds.edge
        cell = qmesh.ds.cell
        bc = qmesh.barycenter()
        for j, d in enumerate(data):
            S = vem.project_to_smspace(d)
            pdata = d.copy()
            edata = np.sum(d[edge], axis=1)/2.0
            cdata = S.value(bc)
            data[j] = np.r_[pdata, edata, cdata]
        qmesh.uniform_refine()
        pmesh = PolygonMesh.from_mesh(qmesh)
        vem.reinit(pmesh)

# error analysis
errorType = ['$\| u - \Pi^\\nabla u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\\nabla u_h\|$'
             ]
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
for i in range(maxit-1):
    S = vem.project_to_smspace(data[i])
    errorMatrix[0, i] = vem.L2_error(S.value)
    errorMatrix[1, i] = vem.H1_semi_error(S.grad_value)

solution['errorMatrix'] = errorMatrix
solution['Ndof'] = Ndof
f = 'solution.mat'
sio.matlab.savemat(f, solution)

print(Ndof)
print(errorMatrix)

show_error_table(Ndof[:-1], errorType, errorMatrix[:, :-1])

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = qmesh.point[:, 0]
y = qmesh.point[:, 1]
cell = qmesh.ds.cell
tri = np.r_['0', cell[:, [1, 2, 0]], cell[:, [3, 0, 2]]]
s = axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)
fig2.colorbar(s)
plt.show()
