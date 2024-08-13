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
mesh = pde.init_mesh(nx=8, ny=8)

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

    glform = LinearForm(pspace)
    glform.add_boundary_integrator(ScalarNeumannBCIntegrator(pde.neumann, q = p+2))
    G = glform.assembly()

    uh = uspace.function(dim = 2)
    uh[0] = uspace.interpolate(lambda p : pde.solutionu(p)[..., 0])
    uh[1] = uspace.interpolate(lambda p : pde.solutionu(p)[..., 1])
    ph = pspace.function()
    ph[:] = pspace.interpolate(pde.solutionp).reshape(-1)
    while True:
        bform = BilinearForm((uspace, uspace))
        bform.add_domain_integrator(VectorMassIntegrator(c=pde.nonlinear_operator(uh)))
        A = bform.assembly()
        AA = bmat([[A, B], [B.T, None]], format='csr', dtype=np.float64)
        FF = np.hstack((F, G[:-1]))

        print("ppp0 : ", np.max(np.abs(A@uh.flatten() + B@ph[:-1] - F)))
        print("ppp1 : ", np.max(np.abs(B.T@uh.flatten() - G[:-1])))

        val = spsolve(AA, FF)

        uhval = val[:uspace.number_of_global_dofs()*2].reshape(2, -1)
        phval = val[uspace.number_of_global_dofs()*2:]

        flag = np.max(np.abs(uhval-uh[:])) < 1e-2
        uh[:] = uhval
        ph[:-1] = phval
        if flag:
            break
    error0 = mesh.error(pde.solutionu, uh)
    error1 = mesh.error(pde.solutionp, ph)
    print("error0:", error0)
    print("error1:", error1)

    if i < maxit-1:
        mesh.uniform_refine()

print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
