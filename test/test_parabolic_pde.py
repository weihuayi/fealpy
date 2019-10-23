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
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.fem.integral_alg import IntegralAlg

box = [-1, 1, -1, 1]
t = 0

p = 1
n = 4

pde = SpaceMeasureDiracSourceData
mesh = pde.init_mesh(n)
node = mesh.node
bc = mesh.entity_barycenter('cell')
integrator = mesh.integrator(p+2)
cellmeasure = mesh.entity_measure('cell')
integralalg = IntegralAlg(integrator, mesh, cellmeasure)

space = LagrangeFiniteElementSpace(mesh, p, spacetype='C')
#uI = space.interpolation(lambda x: pde.solution(x, t))
uI = pde.solution(node,p=node, t=t)
gu = pde.gradient(bc,p=bc,t=t)
uI = space.function()
guI = uI.grad_value(bc)
eta = integralalg.L2_error(gu, guI)
print('eta', eta)



