import argparse

import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat, csc_matrix

from fealpy.pde.poisson_2d import CosCosData

from fealpy.mesh.triangle_mesh import TriangleMesh 

from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace

from fealpy.fem.vector_mass_integrator import VectorMassIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem import MixedBilinearForm, ScalarSourceIntegrator
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.scalar_neumann_bc_integrator import ScalarNeumannBCIntegrator
from fealpy.solver.gamg_solver import GAMGSolver

from fealpy.fem import VectorDarcyIntegrator, ScalarNeumannBCIntegrator
from fealpy.pde.nonlinear_darcy_pde_2d import Data0
import numpy as np
import time
from scipy.sparse import csr_matrix

def Solve(A, b):
    solver = GAMGSolver()
    solver.setup(A)
    x = solver.solve(b)
    return x

class DarcyPreconditionSolver:
    def __init__(self, pde, p=1):
        self.pde = pde
        self.p = p

    def get_IQ_matrix(self):
        pass

    def get_IV_matrix(self):
        pass

    def run(self, n):
        pde = self.pde
        p   = self.p

        mesh = pde.init_mesh(nx=n, ny=n)

        uspace = LagrangeFESpace(mesh, p = p-1, spacetype = 'D', doforder = 'sdofs')
        pspace = LagrangeFESpace(mesh, p = p, spacetype = 'C', doforder = 'sdofs')

        mixform = MixedBilinearForm((pspace, ), (uspace, uspace))
        mixform.add_domain_integrator(VectorDarcyIntegrator()) 
        B = mixform.assembly().tocsc()
        B = self._remove_row(B)

        lform = LinearForm((uspace, uspace))
        lform.add_domain_integrator(VectorSourceIntegrator(f = pde.source, q = p+2))
        F = lform.assembly()

        lform = LinearForm(pspace)
        lform.add_domain_integrator(ScalarSourceIntegrator(lambda p : 1, q = p+2))
        Int = lform.assembly()

        glform = LinearForm(pspace)
        glform.add_boundary_integrator(ScalarNeumannBCIntegrator(pde.neumann, q = p+2))
        G = glform.assembly()
        G = G[:-1]

        uh = uspace.function(dim = 2)
        ph = pspace.function()
        while True:
            bform = BilinearForm((uspace, uspace))
            bform.add_domain_integrator(VectorMassIntegrator(c=pde.nonlinear_operator(uh)))
            A = bform.assembly()

            AA = bmat([[A, B], [B.T, None]], format='csr', dtype=np.float64)
            FF = np.hstack((F, G))

            # lform = LinearForm((uspace, uspace))
            # lform.add_domain_integrator(VectorSourceIntegrator(f = pde.nonlinear_operator0(uh), q = p+2))
            # FFF = lform.assembly()
            # print("ppp0 : ", np.max(np.abs(A@uh.flatten() + B@ph[:-1]- F)))
            # print("ppp1 ppp1: ", np.max(np.abs(B.T@uh.flatten() - G)))

            val = spsolve(AA, FF)

            uhval = val[:uspace.number_of_global_dofs()*2].reshape(2, -1)
            phval = val[uspace.number_of_global_dofs()*2:]

            flag = np.max(np.abs(uhval-uh[:])) < 1e-8
            uh[:] = uhval
            ph[:-1] = phval
            break
            if flag:
                break
        ph[:] -= Int@ph/4

        error0 = mesh.error(pde.solutionu, uh)
        error1 = mesh.error(pde.solutionp, ph, q = p+2)
        return error0, error1

    def _remove_row(self, matrix):
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

n = 2
maxit = 1
pde = Data0() 
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
solver = DarcyPreconditionSolver(pde, 1)
for i in range(maxit):
    errorMatrix[:, i] = solver.run(n)
    n *= 2

print('error:', errorMatrix)
print('d_error:', errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

