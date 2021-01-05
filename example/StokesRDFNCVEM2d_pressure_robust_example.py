#!/usr/bin/env python3
# 
import sys
import numpy as np

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


# FEALPy
## mesh
from fealpy.mesh import TriangleMesh, PolygonMesh, MeshFactory

## space
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import DivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ReducedDivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d

p = 2
tmesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=2, ny=2, meshtype='tri')

# RT 空间
space0 = RaviartThomasFiniteElementSpace2d(tmesh, p=1)


# 三角形转化为多边形网格， 注意这里要求三角形的 0 号边转化后仍然是多边形的 0
# 号边
NC = tmesh.number_of_cells()
NV = 3
node = tmesh.entity('node')
cell = tmesh.entity('cell')[:, [1, 2, 0]]
cellLocation = np.arange(0, (NC+1)*NV, NV)
pmesh = PolygonMesh(node, cell.reshape(-1), cellLocation)

# 缩减虚单元空间
space1 = ReducedDivFreeNonConformingVirtualElementSpace2d(pmesh, p=p)


RT = space1.interpolation_RT(space0)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node(axes, showindex=True)
tmesh.find_edge(axes, showindex=True)
tmesh.find_cell(axes, showindex=True)

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
pmesh.find_node(axes, showindex=True)
pmesh.find_edge(axes, showindex=True)
pmesh.find_cell(axes, showindex=True)

plt.show()





