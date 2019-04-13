#!/usr/bin/env python3
#
"""
This script include the test code of the adaptive VEM for the simplified
friction problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.sfc_2d import SFCModelData2
from fealpy.vem.SFCVEMModel2d import SFCVEMModel2d

pde = SFCModelData2()
qtree = pde.init_mesh(n=2, meshtype='quadtree')
pmesh = qtree.to_pmesh()

vem = SFCVEMModel2d(pde, pmesh, p=1, q=4)
vem.solve(rho=0.1, maxit=40000)

fig = plt.figure()

uI = vem.space.interpolation(pde.solution)

e = np.max(np.abs(uI - vem.uh))
print("The max error:", e)
print(uI)
print(vem.uh)

axes = fig.gca()
pmesh.add_plot(axes)
plt.show()
