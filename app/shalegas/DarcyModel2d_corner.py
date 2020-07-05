#!/usr/bin/env python3
# 
import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace.femdof import multi_index_matrix2d
from pde_data import LeftRightData



n = int(sys.argv[1])
p = int(sys.argv[2])

pde = LeftRightData()
mesh = pde.init_mesh(n=n, meshtype='tri')
space = RaviartThomasFiniteElementSpace2d(mesh, p=p)

udof = space.number_of_global_dofs()
pdof = space.smspace.number_of_global_dofs()
gdof = udof + pdof

uh = space.function()
ph = space.smspace.function()
A = space.stiff_matrix()
B = space.div_matrix()
F1 = space.source_vector(pde.source)
AA = bmat([[A, -B], [-B.T, None]], format='csr')

isBdDof = space.set_dirichlet_bc(uh, pde.neumann)
x = np.r_['0', uh, ph] 
isBdDof = np.r_['0', isBdDof, np.zeros(pdof, dtype=np.bool_)]

FF = np.r_['0', np.zeros(udof, dtype=np.float64), F1]

FF -= AA@x
bdIdx = np.zeros(gdof, dtype=np.int)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof)
T = spdiags(1-bdIdx, 0, gdof, gdof)
AA = T@AA@T + Tbd
FF[isBdDof] = x[isBdDof]
x[:] = spsolve(AA, FF)
uh[:] = x[:udof]
ph[:] = x[udof:]

area = mesh.entity_measure('cell')
x[udof:] -= sum(x[udof:]*area)/sum(area)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor=x[udof:], showcolorbar=True)
node = mesh.entity('node')
bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
ps = mesh.bc_to_point(bc)
V = uh.value(bc)
axes.quiver(ps[:, 0], ps[:, 1], V[:, 0], V[:, 1], angles='xy', units='xy')
axes.set_axis_on()
plt.show()
