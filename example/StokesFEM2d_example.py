
import argparse
import sys
import numpy as np

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
## Stokes model
from fealpy.pde.stokes_model_2d import StokesModelData_0 as PDE

## error anlysis tool
from fealpy.tools import showmultirate


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形或四面体网格上求解 Stokes 问题
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格加密的次数, 默认初始加密 4 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

degree = args.degree
dim = args.dim
nrefine = args.nrefine
maxit = args.maxit


pde = PDE()
mesh = MF.boxmesh2d(pde.box, nx=10, ny=10, meshtype='tri')

errorType = ['$|| u - u_h||_0$',
             '$|| p - p_h||_0$'
             ]

errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    NC = mesh.number_of_cells()

    uspace = LagrangeFiniteElementSpace(mesh, p=degree, spacetype='C')
    pspace = LagrangeFiniteElementSpace(mesh, p=degree-1, spacetype='D')

    ugdof = uspace.number_of_global_dofs()
    pgdof = pspace.number_of_global_dofs()


    uh = uspace.function(dim=dim)
    ph = pspace.function()

    A = uspace.stiff_matrix()
    B0, B1 = uspace.div_matrix(pspace)
    F = uspace.source_vector(pde.source, dim=dim)
    
    

    AA = bmat([[A, None, -B0], [None, A, -B1], [-B0.T, -B1.T, None]], format='csr')
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]

    uspace.set_dirichlet_bc(uh, pde.dirichlet)
    isBdDof = uspace.is_boundary_dof()
    x = np.block([uh.T.flat, ph])
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(pgdof, dtype=np.bool)])

    gdof = 2*ugdof + pgdof
    FF -= AA@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    AA = T@AA@T + Tbd
    FF[isBdDof] = x[isBdDof]
    x[:] = spsolve(AA, FF)
    uh[:, 0] = x[:ugdof]
    uh[:, 1] = x[ugdof:2*ugdof]
    ph[:] = x[2*ugdof:]

    NDof[i] =  gdof 


    errorMatrix[0, i] = uspace.integralalg.error(pde.velocity, uh)
    errorMatrix[1, i] = pspace.integralalg.error(pde.pressure, ph)
    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
showmultirate(plt, 0, NDof, errorMatrix, errorType)
plt.show()



























