#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt

# solver
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import csr_matrix, spdiags, eye, bmat

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import FirstNedelecFiniteElementSpace2d 

# from fealpy.boundarycondition import DirichletBC  #导入边界条件包
from fealpy.fem import DirichletBC # 处理边界条件

from fealpy.pde.MaxwellPDE_2d import SinData as PDE


pde = PDE()
maxit = 6
errorType = ['$|| E - E_h||_{\Omega,0}$']
errorMatrix = np.zeros(maxit)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    mesh = pde.init_mesh(2**i)
    space = FirstNedelecFiniteElementSpace2d(mesh)

    gdof = space.dof.number_of_global_dofs()
    NDof[i] = gdof

    bc = DirichletBC(space, pde.dirichlet) 

    M = space.mass_matrix()
    A = space.curl_matrix()
    b = space.source_vector(pde.source)
    B = A-M 

    Eh = space.function()
    #B, b = bc.apply(B, b, Eh)
    isDDof = space.set_dirichlet_bc(pde.dirichlet, Eh)
    b[isDDof] = Eh[isDDof]

    bdIdx = np.zeros(B.shape[0], dtype=np.int_)
    bdIdx[isDDof] = 1
    Tbd = spdiags(bdIdx, 0, B.shape[0], B.shape[0])
    T = spdiags(1-bdIdx, 0, B.shape[0], B.shape[0])
    B = T@B + Tbd

    Eh[:] = spsolve(B, b)
    # 计算误差
    errorMatrix[i] = space.integralalg.error(pde.solution, Eh)
    print(errorMatrix[i])

print('ratio')
for i in range(1,maxit):
    print(errorMatrix[i-1]/errorMatrix[i])

