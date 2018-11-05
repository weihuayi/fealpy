import sys

import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.surface_poisson_model_3d import SphereSinSinSinData
from fealpy.fem.SurfacePoissonFEMModel import SurfacePoissonFEMModel
from fealpy.mesh.adaptive_tools import AdaptiveMarker
from fealpy.quadrature import TriangleQuadrature
from fealpy.tools.show import showmultirate
from fealpy.mesh.tree_data_structure import Tritree
from fealpy.recovery import FEMFunctionRecoveryAlg
import mpl_toolkits.mplot3d as a3
import pylab as pl




p = int(sys.argv[1])
theta = 0.2
integrator = TriangleQuadrature(6)
ralg = FEMFunctionRecoveryAlg()
pde = SphereSinSinSinData()
mesh = pde.init_mesh(3)
tmesh = Tritree(mesh.node, mesh.ds.cell, irule=1)
pmesh = tmesh.to_conformmesh()
fig0 = pl.figure()
axes0 = a3.Axes3D(fig0)
pmesh.add_plot(axes0)
pl.show()
fem = SurfacePoissonFEMModel(pmesh, pde, p, integrator)
fem.solve()
uh = fem.uh
print(uh)
rguh = ralg.harmonic_average(uh)
eta = fem.recover_estimate(rguh)
tmesh.refine(marker=AdaptiveMarker(eta, theta=theta))
pmesh = tmesh.to_conformmesh()
fem = SurfacePoissonFEMModel(pmesh, pde, p, integrator)
fem.solve()
uh = fem.uh
print(uh)
fig1 = pl.figure()
axes1 = a3.Axes3D(fig1)
pmesh.add_plot(axes1)
pl.show()
















