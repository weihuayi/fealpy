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
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import DivFreeNonConformingVirtualElementSpace2d
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

errorType = ['$|| u - \Pi \\tilde u_h||_0$',
             '$|| p - \\tilde p_h||_0$',
             '$|| \\varepsilon(u) - \\varepsilon(\Pi \\tilde u_h) ||_0$',
             '$|| u - \Pi u_h||_0$',
             '$|| p - p_h||_0$',
             ]
errorMatrix = np.zeros((5, maxit), dtype=np.float)
dof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    #mesh = pde.init_mesh(n=i+2, meshtype='poly') 
    mesh = pde.init_mesh(n=i+2, meshtype='quad') 
    mesh = PolygonMesh.from_mesh(mesh)

    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    idof = (p-2)*(p-1)//2

    dof[i] = NC

    # 完全非协调向量虚单元空间
    uspace = DivFreeNonConformingVirtualElementSpace2d(mesh, p, q=6)
    pspace = ScaledMonomialSpace2d(mesh, p-1)
    uh = uspace.function()
    ph = pspace.function()

    # 缩减非协调向量虚单元空间 
    tuspace = ReducedDivFreeNonConformingVirtualElementSpace2d(mesh, p, q=6)
    # 分片常数压力空间
    tpspace = ScaledMonomialSpace2d(mesh, 0)

    isBdDof = tuspace.boundary_dof()

    tudof = tuspace.number_of_global_dofs()
    tpdof = tpspace.number_of_global_dofs()

    tuh = tuspace.function()
    tph = tpspace.function()

    A = tuspace.stiff_matrix()
    P = tuspace.div_matrix()
    F = tuspace.source_vector(pde.source)
    AA = bmat([[A, P.T], [P, None]], format='csr')
    FF = np.block([F, np.zeros(tpdof, dtype=tuspace.ftype)])

    tuspace.set_dirichlet_bc(tuh, pde.dirichlet)
    x = np.block([tuh, tph])
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(NC*idof+tpdof, dtype=np.bool_)])
    gdof = tudof + tpdof
    FF -= AA@x
    bdIdx = np.zeros(gdof, dtype=np.int)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    AA = T@AA@T + Tbd

    FF[isBdDof] = x[isBdDof]

    x[:] = spsolve(AA, FF)
    tuh[:] = x[:tudof]
    tph[:] = x[tudof:]

    tuspace.project_to_complete_space(pde.source, tuh, tph, uh, ph)
    up = uspace.project_to_smspace(uh)
    tup = tuspace.project_to_smspace(tuh)

    integralalg = uspace.integralalg
    area = sum(mesh.entity_measure('cell'))

    def strain(x, index):
        val = tup.grad_value(x, index)
        return (val + val.swapaxes(-1, -2))/2


    errorMatrix[0, i] = integralalg.L2_error(pde.velocity, tup)

    iph = integralalg.integral(tph)/area
    def tpressure(x, index):
        return tph.value(x, index) - iph
    errorMatrix[1, i] = integralalg.L2_error(pde.pressure, tpressure)

    errorMatrix[2, i] = integralalg.L2_error(pde.strain, strain)
    errorMatrix[3, i] = integralalg.L2_error(pde.velocity, up)

    iph = integralalg.integral(ph)/area
    def pressure(x, index):
        return ph.value(x, index) - iph
    errorMatrix[4, i] = integralalg.L2_error(pde.pressure, pressure)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
showmultirate(plt, 0, dof, errorMatrix, errorType)
plt.show()
