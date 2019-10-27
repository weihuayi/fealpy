#########################################################################
# File Name: test_parabolic_pde.py
# Author: liao
# mail: 1822326109@qq.com 
# Created Time: Wed 23 Oct 2019 11:35:45 AM CST
#########################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.parabolic_model_2d import SpaceMeasureDiracSourceData
from fealpy.pde.parabolic_model_2d import TimeMeasureDiracSourceData

from fealpy.mesh import TriangleMesh
from fealpy.mesh.adaptive_tools import mark
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace

box = [-1, 1, -1, 1]
t = 0.7561523

p = 1
n = 3

#pde = SpaceMeasureDiracSourceData()
pde = TimeMeasureDiracSourceData()
mesh = pde.init_mesh(n)
node = mesh.entity('node')
cell = mesh.entity('cell')
print('cell', cell.shape)
bc = mesh.entity_barycenter('cell')
integrator = mesh.integrator(p+2)
cellmeasure = mesh.entity_measure('cell')

space = LagrangeFiniteElementSpace(mesh, p, spacetype='C')
integralalg = FEMeshIntegralAlg(integrator, mesh)
uI = space.interpolation(lambda x: pde.solution(x, t))
#uI = pde.solution(node,p=node, t=t)
gu = pde.gradient
#uI = space.function()
guI = uI.grad_value
eta = integralalg.L2_error(gu, guI, celltype=True) # size = (cell.shape[0], 2)
eta = np.sum(eta,axis=-1)
print('eta', eta.shape)

tmesh = TriangleMesh(node, cell)
mark = mark(eta, 0.75)
options = tmesh.adaptive_options(method='mean', maxrefine=10,maxcoarsen=0, theta=0.5)
adaptmesh = tmesh.adaptive(eta, options)
#node = adaptmesh.node
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()
