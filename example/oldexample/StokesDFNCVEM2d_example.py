#!/usr/bin/env python3
# 
import sys
import numpy as np

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


# FEALPy
## mesh
from fealpy.mesh import PolygonMesh

## space
from fealpy.functionspace import DivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d

## Stokes model
from fealpy.pde.stokes_model_2d import StokesModelData_0, StokesModelData_1
from fealpy.pde.stokes_model_2d import StokesModelData_2, StokesModelData_3

## error anlysis tool
from fealpy.tools import showmultirate

m = int(sys.argv[1])
p = int(sys.argv[2])
maxit = int(sys.argv[3])

if m == 0:
    pde = StokesModelData_0()
elif m == 1:
    pde = StokesModelData_1()
elif m == 2:
    pde = StokesModelData_2()
elif m == 3:
    pde = StokesModelData_3()

errorType = ['$||  u - \Pi  u_h||_0$',
             '$|| p - p_h||_0$',
             '$|| \\varepsilon(u) - \\varepsilon(\Pi u_h) ||_0$'
             ]
errorMatrix = np.zeros((3, maxit), dtype=np.float)
dof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    mesh = pde.init_mesh(n=i+2, meshtype='quad') 
    mesh = PolygonMesh.from_mesh(mesh)

    NC = mesh.number_of_cells()
    dof[i] = NC

    uspace = DivFreeNonConformingVirtualElementSpace2d(mesh, p, q=6)
    pspace = ScaledMonomialSpace2d(mesh, p-1)

    isBdDof = uspace.boundary_dof()

    udof = uspace.number_of_global_dofs()
    pdof = pspace.number_of_global_dofs()

    uh = uspace.function()
    ph = pspace.function()

    A = uspace.matrix_A()
    P = uspace.matrix_P()
    F = uspace.source_vector(pde.source)

    AA = bmat([[A, P.T], [P, None]], format='csr')
    FF = np.block([F, np.zeros(pdof, dtype=uspace.ftype)])

    uspace.set_dirichlet_bc(uh, pde.dirichlet)
    x = np.block([uh.T.flat, ph])
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(pdof, dtype=np.bool_)])

    gdof = 2*udof + pdof
    FF -= AA@x
    bdIdx = np.zeros(gdof, dtype=np.int)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    AA = T@AA@T + Tbd
    FF[isBdDof] = x[isBdDof]
    x[:] = spsolve(AA, FF)
    uh[:, 0] = x[:udof]
    uh[:, 1] = x[udof:2*udof]
    ph[:] = x[2*udof:]

    up = uspace.project_to_smspace(uh)
    integralalg = uspace.integralalg

    def strain(x, index):
        val = up.grad_value(x, index)
        return (val + val.swapaxes(-1, -2))/2

    iph = integralalg.integral(ph)
    def pressure(x, index):
        return ph.value(x, index) - iph


    errorMatrix[0, i] = integralalg.L2_error(pde.velocity, up)
    errorMatrix[1, i] = integralalg.L2_error(pde.pressure, pressure)
    errorMatrix[2, i] = integralalg.L2_error(pde.strain, strain)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
showmultirate(plt, 0, dof, errorMatrix, errorType)
plt.show()
