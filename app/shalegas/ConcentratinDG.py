#!/usr/bin/env python3
# 
"""

Notes
-----

给定一个求解区域， 区域中有一个固定的流场 v 和一个初始浓度 c， 计算浓度随时间的
变化。
"""
import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh, MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace.femdof import multi_index_matrix2d

from pde_data import  LeftRightData

# 用混合元计算流场
pde = LeftRightData() # 流场从左到右
mesh = pde.init_mesh(n=n, meshtype='tri')
space = RaviartThomasFiniteElementSpace2d(mesh, p=p)

udof = space.number_of_global_dofs()
pdof = space.smspace.number_of_global_dofs()
gdof = udof + pdof + 1

uh = space.function()
ph = space.smspace.function()
A = space.stiff_matrix()
B = space.div_matrix()
C = space.smspace.cell_mass_matrix()[:, 0, :].reshape(-1)
F1 = space.source_vector(pde.source)

AA = bmat([[A, -B, None], [-B.T, None, C[:, None]], [None, C, None]], format='csr')

isBdDof = space.set_dirichlet_bc(uh, pde.neumann)

x = np.r_['0', uh, ph, 0] 
isBdDof = np.r_['0', isBdDof, np.zeros(pdof+1, dtype=np.bool_)]
FF = np.r_['0', np.zeros(udof, dtype=np.float64), F1, 0]

FF -= AA@x
bdIdx = np.zeros(gdof, dtype=np.int)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof)
T = spdiags(1-bdIdx, 0, gdof, gdof)
AA = T@AA@T + Tbd
FF[isBdDof] = x[isBdDof]
x[:] = spsolve(AA, FF)
uh[:] = x[:udof]
ph[:] = x[udof:-1]

# 给定初始的浓度 c， 模拟在 c 在流场中的变化 
# 浓度用间断元表示
