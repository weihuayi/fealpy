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

from fealpy.pde.stokes_model_2d import StokesModelData_0, StokesModelData_1


p = 2
n = 5
tmesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=5, ny=5, meshtype='tri')

# RT 空间
space0 = RaviartThomasFiniteElementSpace2d(tmesh, p=p-1)


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

b0 = space1.source_vector(pde.source)
b1 = space1.pressure_robust_source_vector(pde.source, space0)

plt.show()




