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
from fealpy.functionspace import ReducedDivFreeNonConformingVirtualElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d

## Stokes model
from fealpy.pde.stokes_model_2d import StokesModelData_0, StokesModelData_1
from fealpy.pde.stokes_model_2d import StokesModelData_2, StokesModelData_3
from fealpy.pde.stokes_model_2d import StokesModelData_4, StokesModelData_5
from fealpy.pde.stokes_model_2d import StokesModelData_6

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
elif m == 4:
    pde = StokesModelData_4()
elif m == 5:
    pde = StokesModelData_5()
elif m == 6:
    pde = StokesModelData_6()

errorType = ['$||  u - \Pi  u_h||_0$',
             '$|| p - p_h||_0$',
             '$|| \\varepsilon(u) - \\varepsilon(\Pi u_h) ||_0$'
             ]
errorMatrix = np.zeros((3, maxit), dtype=np.float)
dof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    #mesh = pde.init_mesh(n=i+2, meshtype='poly') 
    mesh = pde.init_mesh(n=i+2, meshtype='quad') 
    mesh = PolygonMesh.from_mesh(mesh)

    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    idof = (p-2)*(p-1)//2

    dof[i] = NC

    uspace = ReducedDivFreeNonConformingVirtualElementSpace2d(mesh, p, q=6)
    pspace = ScaledMonomialSpace2d(mesh, 0)

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
    x = np.block([uh, ph])
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(NC*idof+pdof, dtype=np.bool)])
    gdof = udof + pdof
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

    up = uspace.project_to_smspace(uh)
    integralalg = uspace.integralalg

    def strain(x, index):
        val = up.grad_value(x, index)
        return (val + val.swapaxes(-1, -2))/2

    area = mesh.entity_measure('cell')
    iph = sum(ph*area)/sum(area)
    print("1:", iph)
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
