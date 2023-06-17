#!/usr/bin/env python3
# 

import time
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import FirstNedelecFiniteElementSpace3d 
from fealpy.decorator import cartesian, barycentric

from fealpy.tools.show import showmultirate, show_error_table

from fealpy.boundarycondition import DirichletBC  #导入边界条件包
# solver
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import csr_matrix, spdiags, eye, bmat

#from fealpy.pde.MaxwellPDE import XXX3dData as PDE
#from fealpy.pde.MaxwellPDE import Sin3dData as PDE
#from fealpy.pde.MaxwellPDE import Bubble3dData as PDE
from fealpy.pde.MaxwellPDE import SinData as PDE

import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pde = PDE()
maxit = 5
errorType = ['$|| E - E_h||_{\Omega,0}$']
errorMatrix = np.zeros((1, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    mesh = pde.init_mesh(2**i)
    space = FirstNedelecFiniteElementSpace3d(mesh)

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
    errorMatrix[0, i] = space.integralalg.error(pde.solution, Eh)

showmultirate(plt, 2, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
print(errorMatrix)
plt.show()

fname = "Ord.png"
plt.show()
plt.savefig(fname, dpi=400)
