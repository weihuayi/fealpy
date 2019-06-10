#!/usr/bin/env python3
#
"""
This script include the test code of the adaptive VEM for the simplified
friction problem.

Note
----
1. 质量矩阵的稳定项
1. 右端计算是否合适
"""

import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.sfc_2d import SFCModelData1
from fealpy.vem.SFCVEMModel2d import SFCVEMModel2d

pde = SFCModelData1()
qtree = pde.init_mesh(n=6, meshtype='quadtree')
pmesh = qtree.to_pmesh()

vem = SFCVEMModel2d(pde, pmesh, p=1, q=4)
vem.solve(rho=0.05, maxit=40000)

uI = vem.space.interpolation(pde.solution)

node = pmesh.entity('node')
x = node[:, 0]
y = node[:, 1]
tri = qtree.leaf_cell(celltype='tri')

fig0 = plt.figure()
fig0.set_facecolor('white')
axes = fig0.gca(projection='3d')
axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)

fig1 = plt.figure()
fig1.set_facecolor('white')
axes = fig1.gca(projection='3d')
axes.plot_trisurf(x, y, tri, uI[:len(x)], cmap=plt.cm.jet, lw=0.0)

fig1 = plt.figure()
axes = fig1.gca()
pmesh.add_plot(axes)
plt.show()
