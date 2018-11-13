import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData
from fealpy.fem.SurfacePoissonFEMModel import SurfacePoissonFEMModel
from fealpy.mesh.adaptive_tools import mark 
from fealpy.quadrature import TriangleQuadrature
from fealpy.tools.show import showmultirate
from fealpy.mesh.tree_data_structure import Tritree
from fealpy.recovery import FEMFunctionRecoveryAlg
import mpl_toolkits.mplot3d as a3
import pylab as pl

class AdaptiveMarker():
    def __init__(self, eta, theta=0.2, ctheta=0.1):
        self.eta = eta
        self.theta = theta
        self.ctheta = ctheta

    def refine_marker(self, tree):
        leafCellIdx = tree.leaf_cell_index()
        NC = tree.number_of_cells()
        eta = np.zeros(NC, dtype=tree.ftype)
        idxmap = tree.celldata['idxmap']
        np.add.at(eta, idxmap, self.eta) 
        markedIdx = mark(eta[leafCellIdx], self.theta)
        return leafCellIdx[markedIdx] 

    def coarsen_marker(self, tree):
        return None 

p = int(sys.argv[1])
theta = 0.1
maxit = 20 
errorType = ['$|| u_I - u_h ||_{l_2}$',
             '$|| u - u_h ||_{S,0}$',
             '$||\\nabla_S u - \\nabla_S u_h||_{S,0}$'
             ]
             

Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
integrator = TriangleQuadrature(3)
ralg = FEMFunctionRecoveryAlg()
pde = SphereSinSinSinData()
mesh = pde.init_mesh(3)
tmesh = Tritree(mesh.node, mesh.ds.cell, irule=1)
pmesh = tmesh.to_conformmesh()

fig = pl.figure()
axes = a3.Axes3D(fig)
pmesh.add_plot(axes)

for i in range(maxit):
    print('step:', i)
    fem = SurfacePoissonFEMModel(pmesh, pde, p, integrator)
    fem.solve()
    uh = fem.uh
    rguh = ralg.harmonic_average(uh)
    eta = fem.recover_estimate(rguh)
    Ndof[i] = len(fem.uh)
    errorMatrix[0, i] = fem.get_l2_error()
    errorMatrix[1, i] = fem.get_L2_error()
    errorMatrix[2, i] = fem.get_H1_semi_error()
    if i < maxit - 1:
        tmesh.refine(marker=AdaptiveMarker(eta, theta=theta))
        pmesh = tmesh.to_conformmesh()
        fig = pl.figure()
        axes = a3.Axes3D(fig)
        pmesh.add_plot(axes)
pl.show()
print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 6, Ndof, errorMatrix, errorType)
plt.show()
