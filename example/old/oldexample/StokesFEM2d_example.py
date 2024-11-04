'''!
@file StokesFEM2d_example.py
@author wangpengxiang
@date 11/04/2021
@brief Stokes二维程序代码
'''

import argparse
import sys
import numpy as np

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
from fealpy.pde.stokes_model_2d import StokesModelData_6 as PDE

from fealpy.decorator import cartesian,barycentric
from fealpy.tools import showmultirate

degree = 2
dim = 2
maxit = 4
ns = 15
pde = PDE()
mesh = pde.init_mesh(n=2)
errorType = ['$|| u - u_h||_0$',
             '$|| p - p_h||_0$'
             ]

errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

ctx = DMumpsContext()
ctx.set_silent()

for i in range(maxit):
    print("The {}-th computation:".format(i))

    uspace = LagrangeFiniteElementSpace(mesh, p=degree)
    pspace = LagrangeFiniteElementSpace(mesh, p=degree-1)

    ugdof = uspace.number_of_global_dofs()
    pgdof = pspace.number_of_global_dofs()

    uh = uspace.interpolation(pde.velocity)
    ph = pspace.function()

    A = 1/2*uspace.stiff_matrix()
    B0, B1 = uspace.div_matrix(pspace)
    F = uspace.source_vector(pde.source, dim=dim)    
    C = pspace.mass_matrix()
    
    qf = mesh.integrator(4, 'cell')
    bcs,ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')

    ## 速度空间
    uphi = uspace.basis(bcs)
    ugphi = uspace.grad_basis(bcs)
    ucell2dof = uspace.cell_to_dof()
    
    AA = bmat([[A, None,B0], [None, A, B1], [B0.T, B1.T, None]], format='csr')
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]
    
    isBdDof = uspace.is_boundary_dof()
    gdof = 2*ugdof + pgdof
    x = np.zeros(gdof,np.float)
    ipoint = uspace.interpolation_points()
    uso = pde.dirichlet(ipoint)
    x[0:ugdof][isBdDof] = uso[:,0][isBdDof]
    x[ugdof:2*ugdof][isBdDof] = uso[isBdDof][:,1]
   
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(pgdof, dtype=np.bool_)])

    FF -= AA@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    AA = T@AA@T + Tbd
    FF[isBdDof] = x[isBdDof]

    ctx.set_centralized_sparse(AA)
    xx = FF.copy()
    ctx.set_rhs(xx)
    ctx.run(job=6)
    uh[:, 0] = xx[:ugdof]
    uh[:, 1] = xx[ugdof:2*ugdof]
    ph[:] = xx[2*ugdof:]

    NDof[i] =  gdof 
    
    uc1 = pde.velocity(mesh.node)
    NN = mesh.number_of_nodes()
    uc2 = uh[:NN]
    up1 = pde.pressure(mesh.node)
    up2 = ph[:NN]
    
    NDof[i] =  gdof 
    area = sum(mesh.entity_measure('cell'))

    iph = pspace.integralalg.integral(ph)/area
    
    ph[:] = ph[:]-iph
 
    errorMatrix[0, i] = uspace.integralalg.error(pde.velocity, uh)
    errorMatrix[1, i] = pspace.integralalg.error(pde.pressure, ph)
    #errorMatrix[0, i] = np.abs(uc1-uc2).max()
    #errorMatrix[1, i] = np.abs(up1-up2).max()
    if i < maxit-1:
        mesh.uniform_refine()
    
ctx.destroy()
print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix, errorType)
plt.show()


























