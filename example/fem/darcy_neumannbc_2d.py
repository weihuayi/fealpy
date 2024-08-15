import argparse

import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, csc_matrix

from fealpy.pde.poisson_2d import CosCosData
from pde import LaplacePDE

from fealpy.mesh.triangle_mesh import TriangleMesh 

from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace

from fealpy.fem.vector_mass_integrator import VectorMassIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem import MixedBilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.scalar_neumann_bc_integrator import ScalarNeumannBCIntegrator

from fealpy.fem import VectorDarcyIntegrator, ScalarNeumannBCIntegrator
from fealpy.pde.nonlinear_darcy_pde_2d import Data0
import numpy as np
import time
from scipy.sparse import csr_matrix

def plot_function(uh, u):
    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')

    NC = mesh.number_of_cells()

    mid = mesh.entity_barycenter("cell")
    node = mesh.entity("node")
    cell = mesh.entity("cell")

    coor = node[cell]
    val = u(mid) 
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], uh[ii]*np.ones(3), color = 'r', lw=0.0)#数值解图像
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], val[ii]*np.ones(3), color = 'b', lw=0.0)
    plt.show()

def plot_linear_function(uh, u):
    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')

    NC = mesh.number_of_cells()

    mid = mesh.entity_barycenter("cell")
    node = mesh.entity("node")
    cell = mesh.entity("cell")

    coor = node[cell]
    val = u(node).reshape(-1) 
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], uh[cell[ii]], color = 'r', lw=0.0)#数值解图像
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], val[cell[ii]], color = 'b', lw=0.0)
    plt.show()

"""
def Solve(A, b):
    from mumps import DMumpsContext
    from scipy.sparse.linalg import minres, gmres

    NN = len(b)
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    x = np.array(b)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy() # Cleanup
    '''
    #x, _ = minres(A, b, x0=b, tol=1e-10)
    x, _ = gmres(A, b, tol=1e-10)
    '''
    return x
"""

def remove_row(matrix):
    # 获取原始数据
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    # 找到最后一列的开始位置
    last_col_start = indptr[-2]

    # 去掉最后一列数据
    new_data = data[:last_col_start]
    new_indices = indices[:last_col_start]
    new_indptr = indptr[:-1].copy()  # 去掉最后一列的列指针

    # 创建新的 CSC 矩阵，移除最后一列
    new_matrix = csc_matrix((new_data, new_indices, new_indptr), shape=(matrix.shape[0], matrix.shape[1] - 1))
    return new_matrix


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=1, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit

pde = Data0() 
mesh = pde.init_mesh(nx=nx, ny=ny)

errorType = ['$||u - u_h||_{\\Omega, 0}$', 
             '$||p - p_h||_{\\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    uspace = LagrangeFESpace(mesh, p = p-1, spacetype = 'D', doforder = 'sdofs')
    pspace = LagrangeFESpace(mesh, p = p, spacetype = 'C', doforder = 'sdofs')
    NDof[i] = uspace.number_of_global_dofs()*2 + pspace.number_of_global_dofs()

    mixform = MixedBilinearForm((pspace, ), (uspace, uspace))
    mixform.add_domain_integrator(VectorDarcyIntegrator()) 
    B = mixform.assembly().tocsc()
    B = remove_row(B)

    lform = LinearForm((uspace, uspace))
    lform.add_domain_integrator(VectorSourceIntegrator(f = pde.source, q = p+2))
    F = lform.assembly()

    lform = LinearForm(pspace)

    glform = LinearForm(pspace)
    glform.add_boundary_integrator(ScalarNeumannBCIntegrator(pde.neumann, q = p+2))
    G = glform.assembly()

    uh = uspace.function(dim = 2)
    uh[0] = uspace.interpolate(lambda p : pde.solutionu(p)[..., 0])
    uh[1] = uspace.interpolate(lambda p : pde.solutionu(p)[..., 1])
    ph = pspace.function()
    ph[:] = pspace.interpolate(pde.solutionp).reshape(-1)
    pI = ph.copy()
    uI = uh.copy()
    ph[:] = 0
    uh[:] = 0
    while True:
        bform = BilinearForm((uspace, uspace))
        bform.add_domain_integrator(VectorMassIntegrator(c=pde.nonlinear_operator(uh)))
        A = bform.assembly()
        AA = bmat([[A, B], [B.T, None]], format='csr', dtype=np.float64)
        FF = np.hstack((F, G[:-1]))
        lform = LinearForm((uspace, uspace))
        lform.add_domain_integrator(VectorSourceIntegrator(f = pde.nonlinear_operator0(uh), q = p+2))
        FFF = lform.assembly()

        #print("ppp : ", np.max(np.abs(A@uh.flatten() - FFF)))
        #print("ppp0 : ", np.max(np.abs(A@uh.flatten() + B@ph[:-1] - F)))
        #print("ppp2 : ", np.max(np.abs(B@ph[:-1])))
        #print("ppp1 : ", np.max(np.abs(B.T@uh.flatten() - G[:-1])))

        val = spsolve(AA, FF)

        uhval = val[:uspace.number_of_global_dofs()*2].reshape(2, -1)
        phval = val[uspace.number_of_global_dofs()*2:]

        flag = np.max(np.abs(uhval-uh[:])) < 1e-4
        uh[:] = uhval
        ph[:-1] = phval
        if flag:
            break
    uhf = uh.flatten()
    uIf = uI.flatten()
    #for i in range(len(uhf)):
    #    print("uh{}: uI: {}, diff: {}".format(uhf[i], uIf[i], uhf[i] - uIf[i]))
    error0 = mesh.error(pde.solutionu, uh)
    error1 = mesh.error(pde.solutionp, ph)
    print("error0:", error0)
    print("error1:", error1)
    #plot_function(uh[0], lambda p : pde.solutionu(p)[..., 0])
    plot_function(uh[1], lambda p : pde.solutionu(p)[..., 1])

    #plot_linear_function(ph, pde.solutionp)


    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
