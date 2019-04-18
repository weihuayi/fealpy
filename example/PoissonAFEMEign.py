#!/usr/bin/env python3
# 

"""
几种计算最小特征值最小特征值方法的比较


Usage
-----

./PoissonAFEMEign.py 0.2 50 0

Test Environment
----------------
System: Ubuntu 18.04.2 LTS 64 bit
Memory: 15.6 GB
Processor: Intel® Core™ i7-4702HQ CPU @ 2.20GHz × 8

Result
------

CrackData:

old method: 24452 nodes
    smallest eigns: 16.746135567145902 with time:  57.65254141800051

hu method: 26265 nodes
    smallest eigns: 16.745923914871994 with time:  14.745063177000702

me method: 24379 nodes
    smallest eigns: 16.746147214383882 with time:  14.566903421999996

LShapeData:
     ./PoissonAFEMEign.py 0.15 50 0

     theta = 0.15
     maxit = 50

    old method: 30148 nodes
        smallest eigns: 9.64071499801269 with time:  51.8588675280007

    hu method: 30158 nodes
        smallest eigns: 9.640716887235763 with time:  16.846451205999983

    me method: 30210 nodes
        smallest eigns: 9.640713070938679 with time:  17.15686360500058
"""

print(__doc__)

import sys
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse import eye, csr_matrix, bmat

from fealpy.pde.poisson_2d import CrackData, LShapeRSinData
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.solver.eigns import picard
from fealpy.recovery import FEMFunctionRecoveryAlg
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.mesh.adaptive_tools import mark
import pyamg

def savesolution(uh, fname):
    """Save the fem solution for matlab plotting

    :uh: FEM solution 
    :fname: file name
    :returns: None

    """
    mesh = uh.space.mesh
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    data = {'node': node, 'elem': cell+1, 'solution': uh}
    sio.matlab.savemat(fname, data)
    pass


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

    # 2. 以 u_H 为右端项自适应求解 -\Deta u = d*u_H

    I = eye(gdof) 
    for i in range(maxit):
        if (step > 0) and (i in idx):
            N = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig('mesh_0_' + str(i) + '_' + str(N) +'.pdf')

        # 恢复型后验误差估计
        uh = space.function(array=uh)
        rguh = ralg.harmonic_average(uh)

        intalg = FEMeshIntegralAlg(integrator, mesh, area)

        def fun(x):
            return (rguh.value(x) - uh.grad_value(x))**2

        eta = intalg.integral(fun, celltype=True)
        eta = np.sqrt(eta.sum(axis=-1))
        markedCell = mark(eta, theta)
        # 自适应
        IM = mesh.bisect(markedCell, returnim=True)

        # 插值
        I = IM@I
        uh = IM@uh

        # 细网格上求解
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

    uh *= u[-1]
    uh += I@u[:-1]
    uh /= np.max(np.abs(uh))
    
    uh = space.function(array=uh)
    return uh

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

    gdof = space.number_of_global_dofs()
    uH = np.zeros(gdof, dtype=np.float)

    A = AH[isFreeHDof, :][:, isFreeHDof].tocsr()
    M = MH[isFreeHDof, :][:, isFreeHDof].tocsr()
    uH[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof))) 

    uh = space.function()
    uh[:] = uH

    # 2. 以 u_H 为右端项自适应求解 -\Deta u = u_H
    I = eye(gdof) 
    for i in range(maxit):
        if (step > 0) and (i in idx):
            N = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig('mesh_1_' + str(i) + '_' + str(N) + '.pdf')

        # 重构型后验误差估计与二分法自适应
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

    u = np.zeros(len(isFreeDof))
    ## 求解特征值
    A = AA[isFreeDof, :][:, isFreeDof].tocsr()
    M = MM[isFreeDof, :][:, isFreeDof].tocsr()
    u[isFreeDof], d = picard(A, M, np.ones(sum(isFreeDof))) 
    end = timer()
    print("smallest eigns:", d, "with time: ", end - start)

    uh *= u[-1]
    uh += I@u[:-1]

    uh /= np.max(np.abs(uh))
    
    uh = space.function(array=uh)
    return uh

def eigns_2(pde, theta, maxit, step=0):
    """
    经典求解自适应求解最小特征值的方法：
    1. 在每层网格上求解最小特征值
    1. 根据特征向量对网格做自适应
    """
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
    A = space.stiff_matrix(integrator, area)
    M = space.mass_matrix(integrator, area)
    isFreeDof = ~(space.boundary_dof())

    gdof = space.number_of_global_dofs()
    uh = np.zeros(gdof, dtype=np.float)

    A = A[isFreeDof, :][:, isFreeDof].tocsr()
    M = M[isFreeDof, :][:, isFreeDof].tocsr()
    uh[isFreeDof], d = picard(A, M, np.ones(sum(isFreeDof))) 


    for i in range(maxit):
        if (step > 0) and (i in idx):
            N = mesh.number_of_nodes()
            fig = plt.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
            mesh.add_plot(axes, cellcolor='w')
            fig.savefig('mesh_2_' + str(i) + '_' + str(N) + '.pdf')

        uh = space.function(array=uh)
        rguh = ralg.harmonic_average(uh)
        intalg = FEMeshIntegralAlg(integrator, mesh, area)
        def fun(x):
            return (rguh.value(x) - uh.grad_value(x))**2
        eta = intalg.integral(fun, celltype=True)
        eta = np.sqrt(eta.sum(axis=-1))
        
        markedCell = mark(eta, theta)
        IM = mesh.bisect(markedCell, returnim=True)
        uh = IM@uh

        space = LagrangeFiniteElementSpace(mesh, 1) 
        gdof = space.number_of_global_dofs()
        print(i, ": ", gdof)

        area = mesh.entity_measure('cell')
        A = space.stiff_matrix(integrator, area)
        M = space.mass_matrix(integrator, area)
        isFreeDof = ~(space.boundary_dof())
        A = A[isFreeDof, :][:, isFreeDof].tocsr()
        M = M[isFreeDof, :][:, isFreeDof].tocsr()
        uh[isFreeDof], d = picard(A, M, uh[isFreeDof]) 

    end = timer()
    print("smallest eigns:", d, "with time: ", end - start)

    uh = space.function(array=uh)
    return uh

def eigns_3(pde, n=10):

    mesh = pde.init_mesh(n=n, meshtype='tri')
    integrator = mesh.integrator(4)

    # 1. 粗网格上求解最小特征值问题
    area = mesh.entity_measure('cell')
    space = LagrangeFiniteElementSpace(mesh, 1) 
    gdof = space.number_of_global_dofs()
    uh = np.zeros(gdof, dtype=np.float)

    A = space.stiff_matrix(integrator, area)
    M = space.mass_matrix(integrator, area)
    isFreeHDof = ~(space.boundary_dof())

    A = A[isFreeHDof, :][:, isFreeHDof].tocsr()
    M = M[isFreeHDof, :][:, isFreeHDof].tocsr()

    uh[isFreeHDof], d = picard(A, M, np.ones(sum(isFreeHDof)))
    print("smallest eigns:", d)
    uh = space.function(array=uh)
    return uh


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
#pde = CrackData()

u0 = eigns_0(pde, theta, maxit, step)
u1 = eigns_1(pde, theta, maxit, step)
u2 = eigns_2(pde, theta, maxit, step)

u0.add_plot(plt)
u1.add_plot(plt)
u2.add_plot(plt)

savesolution(u0, 'u0.mat')
savesolution(u1, 'u1.mat')
savesolution(u2, 'u2.mat')

#plt.show()

