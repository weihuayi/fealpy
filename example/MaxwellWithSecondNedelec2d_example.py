#!/usr/bin/env python3
# 

import time
import sys
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import SecondNedelecFiniteElementSpace2d
from fealpy.functionspace import LagrangeFiniteElementSpace, ScaledMonomialSpace2d
from fealpy.decorator import cartesian, barycentric

from fealpy.tools.show import showmultirate, show_error_table

from fealpy.boundarycondition import DirichletBC #导入边界条件包
# solver
from scipy.sparse import bmat
from fealpy.functionspace import BernsteinFiniteElementSpace
from scipy.sparse.linalg import spsolve, cg

#from fealpy.pde.MaxwellPDE import Sin3dData as PDE
#from fealpy.pde.MaxwellPDE import Bubble3dData as PDE
from fealpy.pde.MaxwellPDE2d import SinData as PDE

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

    mesh = pde.init_mesh(nx=2**i, ny=2**i)
    space = SecondNedelecFiniteElementSpace2d(mesh, p=int(sys.argv[1]))
    gdof = space.dof.number_of_global_dofs()
    NDof[i] = gdof

    bc = DirichletBC(space, pde.dirichlet) 

    M = space.mass_matrix()
    A = space.curl_matrix()
    b = space.source_vector(pde.source)
    B = A-M 

    Eh = space.function()
    B, b = bc.apply(B, b, Eh)
    Eh[:] = spsolve(B, b)
    # 计算误差

    errorMatrix[0, i] = space.integralalg.error(pde.solution, Eh)
    print(errorMatrix)

showmultirate(plt, 1, NDof, errorMatrix,  errorType, propsize=20)
show_error_table(NDof, errorType, errorMatrix)
plt.show()
