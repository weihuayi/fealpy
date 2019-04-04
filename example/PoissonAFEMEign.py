#!/usr/bin/env python3
# 


import sys
from timeit import default_timer as timer

import numpy as np
import scipy.io as sio
from scipy.sparse import eye, csr_matrix, bmat

from fealpy.pde.poisson_2d import CrackData, LShapeRSinData
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.solver.eigns import picard
from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.mesh.adaptive_tools import mark
import pyamg


def eigns_0(pde, theta, maxit, step=0):
    start = timer()

    if step == 0:
        idx = []
    else:
        idx =list(range(0, maxit, step)) + [maxit-1]

    mesh = pde.init_mesh(n=4, meshtype='tri')
    integrator = mesh.integrator(3)
    ralg = FEMFunctionRecoveryAlg()

    # 1. 粗网格上求解最小特征值问题
    area = mesh.entity_measure('cell')
    space = LagrangeFiniteElementSpace(mesh, 1) 
    gdof = space.number_of_global_dofs()
    uh = np.zeros(gdof, dtype=np.float)

    AH = space.stiff_matrix(integrator, area)
    MH = space.mass_matrix(integrator, area)
    isFreeHDof = ~(space.boundary_dof())

    A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
    M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()

    uh[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof)))

    # 2. 以 u_H 为右端项自适应求解 -\Deta u = D*u_H
    I = eye(gdof) 
    for i in range(maxit):
        if (step > 0) and (i in idx):
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig('mesh_0_' + str(i) + '.pdf')

        uh = space.function(array=uh)
        rguh = ralg.harmonic_average(uh)

        intalg = FEMeshIntegralAlg(integrator, mesh, area)

        def fun(x):
            return (rguh.value(x) - uh.grad_value(x))**2
        eta = intalg.integral(fun, celltype=True)
        eta = np.sqrt(eta.sum(axis=-1))
        markedCell = mark(eta, theta)
        IM = mesh.bisect(markedCell, returnim=True)
        I = IM@I
        uh = IM@uh

        space = LagrangeFiniteElementSpace(mesh, 1) 
        gdof = space.number_of_global_dofs()
        print(i, ": ", gdof)

        area = mesh.entity_measure('cell')
        A = space.stiff_matrix(integrator, area)
        M = space.mass_matrix(integrator, area)
        isFreeDof = ~(space.boundary_dof())

        b = d*M@uh

        ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())  
        uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))

    # 3. 把 uh 加入粗网格空间, 组装刚度和质量矩阵

    w0 = uh@A
    w1 = w0@uh
    w2 = w0@I
    AA = bmat([[AH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

    w0 = uh@M
    w1 = w0@uh
    w2 = w0@I
    MM = bmat([[MH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

    isFreeDof = np.r_[isFreeHDof, True]

    u = np.zeros(len(isFreeDof))

    ## 求解特征值
    A = AA[isFreeDof, :][:, isFreeDof].tocsr()
    M = MM[isFreeDof, :][:, isFreeDof].tocsr()
    u[isFreeDof], d = picard(A, M, np.ones(sum(isFreeDof))) 

    end = timer()

    print("smallest eigns:", d, "with time: ", end - start)

def eigns_1(pde, theta, maxit, step=0):
    start = timer()

    if step == 0:
        idx = []
    else:
        idx =list(range(0, maxit, step)) + [maxit-1]

    ralg = FEMFunctionRecoveryAlg()

    mesh = pde.init_mesh(n=4, meshtype='tri')
    integrator = mesh.integrator(3)

    # 1. 粗网格上求解最小特征值问题
    area = mesh.entity_measure('cell')
    space = LagrangeFiniteElementSpace(mesh, 1) 
    AH = space.stiff_matrix(integrator, area)
    MH = space.mass_matrix(integrator, area)
    isFreeHDof = ~(space.boundary_dof())

    A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
    M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
    v, d = picard(A, M, np.ones(sum(isFreeHDof))) 

    gdof = space.number_of_global_dofs()
    uH = np.zeros(gdof, dtype=np.float)
    uH[isFreeHDof] = v 

    uh = space.function()
    uh[:] = uH

    # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
    I = eye(gdof) 
    for i in range(maxit):
        if (step > 0) and (i in idx):
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig('mesh_1_' + str(i) + '.pdf')

        rguh = ralg.harmonic_average(uh)
        intalg = FEMeshIntegralAlg(integrator, mesh, area)
        def fun(x):
            return (rguh.value(x) - uh.grad_value(x))**2
        eta = intalg.integral(fun, celltype=True)
        eta = np.sqrt(eta.sum(axis=-1))
        
        markedCell = mark(eta, theta)
        IM = mesh.bisect(markedCell, returnim=True)
        I = IM@I
        uH = IM@uH

        space = LagrangeFiniteElementSpace(mesh, 1) 
        gdof = space.number_of_global_dofs()
        print(i, ": ", gdof)

        area = mesh.entity_measure('cell')
        A = space.stiff_matrix(integrator, area)
        M = space.mass_matrix(integrator, area)
        isFreeDof = ~(space.boundary_dof())
        b = M@uH

        ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())  
        uh = space.function()
        uh[:] = uH
        uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))

# 3. 把 uh 加入粗网格空间, 组装刚度和质量矩阵

    w0 = uh@A
    w1 = w0@uh
    w2 = w0@I
    AA = bmat([[AH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

    w0 = uh@M
    w1 = w0@uh
    w2 = w0@I
    MM = bmat([[MH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

    isFreeDof = np.r_[isFreeHDof, True]

## 求解特征值
    A = AA[isFreeDof, :][:, isFreeDof].tocsr()
    M = MM[isFreeDof, :][:, isFreeDof].tocsr()
    v, d = picard(A, M, np.ones(sum(isFreeDof))) 
    end = timer()
    print("smallest eigns:", d, "with time: ", end - start)

def eigns_2(pde, theta, maxit, step=0):
    start = timer()

    if step == 0:
        idx = []
    else:
        idx =list(range(0, maxit, step)) + [maxit-1]

    ralg = FEMFunctionRecoveryAlg()

    mesh = pde.init_mesh(n=4, meshtype='tri')
    integrator = mesh.integrator(3)

    # 1. 粗网格上求解最小特征值问题
    area = mesh.entity_measure('cell')
    space = LagrangeFiniteElementSpace(mesh, 1) 
    AH = space.stiff_matrix(integrator, area)
    MH = space.mass_matrix(integrator, area)
    isFreeHDof = ~(space.boundary_dof())

    A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
    M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
    v, d = picard(A, M, np.ones(sum(isFreeHDof))) 

    gdof = space.number_of_global_dofs()
    uH = np.zeros(gdof, dtype=np.float)
    uH[isFreeHDof] = v 

    uh = space.function()
    uh[:] = uH

    # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
    I = eye(gdof) 
    for i in range(maxit):
        if (step > 0) and (i in idx):
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig('mesh_1_' + str(i) + '.pdf')

        rguh = ralg.harmonic_average(uh)
        intalg = FEMeshIntegralAlg(integrator, mesh, area)
        def fun(x):
            return (rguh.value(x) - uh.grad_value(x))**2
        eta = intalg.integral(fun, celltype=True)
        eta = np.sqrt(eta.sum(axis=-1))
        
        markedCell = mark(eta, theta)
        IM = mesh.bisect(markedCell, returnim=True)
        I = IM@I
        uH = IM@uH

        space = LagrangeFiniteElementSpace(mesh, 1) 
        gdof = space.number_of_global_dofs()
        print(i, ": ", gdof)

        area = mesh.entity_measure('cell')
        A = space.stiff_matrix(integrator, area)
        M = space.mass_matrix(integrator, area)
        isFreeDof = ~(space.boundary_dof())
        b = M@uH

        ml = pyamg.ruge_stuben_solver(A[isFreeDof, :][:, isFreeDof].tocsr())  
        uh = space.function()
        uh[:] = uH
        uh[isFreeDof] = ml.solve(b[isFreeDof], x0=uh[isFreeDof], tol=1e-12, accel='cg').reshape((-1,))

# 3. 把 uh 加入粗网格空间, 组装刚度和质量矩阵

    w0 = uh@A
    w1 = w0@uh
    w2 = w0@I
    AA = bmat([[AH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

    w0 = uh@M
    w1 = w0@uh
    w2 = w0@I
    MM = bmat([[MH, w2.reshape(-1, 1)], [w2, w1]], format='csr')

    isFreeDof = np.r_[isFreeHDof, True]

## 求解特征值
    A = AA[isFreeDof, :][:, isFreeDof].tocsr()
    M = MM[isFreeDof, :][:, isFreeDof].tocsr()
    v, d = picard(A, M, np.ones(sum(isFreeDof))) 
    end = timer()
    print("smallest eigns:", d, "with time: ", end - start)
    pass

theta = float(sys.argv[1])
maxit = int(sys.argv[2])
step = int(sys.argv[3])

info = """
theta : %f
maxit : %d
 step : %d
"""%(theta, maxit, step)
print(info)


pde = LShapeRSinData()

eigns_0(pde, theta, maxit, step)
eigns_1(pde, theta, maxit, step)
