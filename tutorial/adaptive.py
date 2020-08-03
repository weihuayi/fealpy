
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
import pyamg

from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate


def residual_estimate(uh, f):
    space = uh.space
    mesh = space.mesh

    NC = mesh.number_of_cells()
    eta = np.zeros(NC, dtype=space.ftype)

    bc = np.array([1/3, 1/3, 1/3], dtype=space.ftype)
    grad = uh.grad_value(bc) # (NC, 2)

    cellmeasure = mesh.entity_measure('cell')
    ch = np.sqrt(cellmeasure)
    edgemeasure = mesh.entity_measure('edge')

    edge2cell = mesh.ds.edge_to_cell() # 
    n = mesh.edge_unit_normal() # (NE, 2)

    # edge2cell[i, 0] == edge2cell[i, 1]
    # (NE, 2) * (NE, 2) --> (NE, )
    J = edgemeasure*np.sum(
            (grad[edge2cell[:, 0]] - grad[edge2cell[:, 1]])*n, axis=-1)**2
    np.add.at(eta, edge2cell[:, 0], J)
    np.add.at(eta, edge2cell[:, 1], J)

    eta *= ch
    eta *= 0.125 
    
    # \int_\tau f**2 dx 
    eta += cellmeasure*space.integralalg.cell_integral(f, power=2) 

    return np.sqrt(eta)

def recovery_estimate(uh):
    space = uh.space
    rguh = space.grad_recovery(uh, method='simple') # (NN, 2) array and a vector finte element function
    eta = space.integralalg.error(rguh.value, uh.grad_value, power=2,
            celltype=True) # (NC, )
    return eta


n = int(sys.argv[1])
mf = MeshFactory()
pde = LShapeRSinData()

mesh = mf.boxmesh2d([-1, 1, -1, 1], nx=n, ny=n, meshtype='tri', 
        threshold=lambda p: (p[..., 0] > 0.0) & (p[..., 1] < 0.0))

mesh.add_plot(plt)
plt.savefig('./test-0.png')
plt.close()

theta = 0.2
maxit = 40

errorType = ['$||u - u_h||_{0}$', '$||\\nabla u - \\nabla u_h||_0$', '$\eta$']
NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)

for i in range(maxit):
    print('step: ', i)
    space = LagrangeFiniteElementSpace(mesh, p=1)

    NDof[i] = space.number_of_global_dofs()
    A = space.stiff_matrix(q=1)
    F = space.source_vector(pde.source)

    bc = DirichletBC(space, pde.dirichlet)
    uh = space.function()

    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)
    #eta = residual_estimate(uh, pde.source)
    eta = recovery_estimate(uh)
    errorMatrix[2, i] = np.sqrt(np.sum(eta**2))

    if i < maxit - 1:
        isMarkedCell = mark(eta, theta=theta)
        mesh.bisect(isMarkedCell)
        mesh.add_plot(plt)
        plt.savefig('./test-' + str(i+1) + '.png')
        plt.close()


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
showmultirate(plt, maxit-5, NDof, errorMatrix, errorType, propsize=20, lw=2,
        ms=4)
plt.show()


