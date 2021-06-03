#!/usr/bin/env python3

import numpy as np
import sys
from scipy.sparse.linalg import spsolve

from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.pde.poisson_2d import ArctanData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.mesh import TriangleMesh


def test_poisson():

    p = 1  # 有限元空间次数, 可以增大 p， 看输出结果的变化
    n = 4  # 初始网格加密次数
    maxit = 4  # 最大迭代次数

    pde = PDE()
    mesh = pde.init_mesh(n=n)

    errorMatrix = np.zeros((2, maxit), dtype=np.float)
    NDof = np.zeros(maxit, dtype=np.float)

    for i in range(maxit):
        space = LagrangeFiniteElementSpace(mesh, p=p)  # 建立有限元空间

        NDof[i] = space.number_of_global_dofs()  # 有限元空间自由度的个数
        bc = DirichletBC(space, pde.dirichlet)  # DirichletBC 条件

        uh = space.function()  # 有限元函数
        A = space.stiff_matrix()  # 刚度矩阵
        F = space.source_vector(pde.source)  # 载荷向量

        A, F = bc.apply(A, F, uh)  # 处理边界条件

        uh[:] = spsolve(A, F).reshape(-1)  # 稀疏矩阵直接解法器

        # ml = pyamg.ruge_stuben_solver(A)  # 代数多重网格解法器
        # uh[:] = ml.solve(F, tol=1e-12, accel='cg').reshape(-1)

        errorMatrix[0, i] = space.integralalg.L2_error(
            pde.solution, uh
        )  # 计算 L2 误差
        errorMatrix[1, i] = space.integralalg.L2_error(
            pde.gradient, uh.grad_value
        )  # 计算 H1 误差

        if i < maxit - 1:
            mesh.uniform_refine()  # 一致加密网格

    assert (errorMatrix < 1.0).all()

class PdeTest():
    def __int__(self):
        pass
    def Arctan(self, p=1, n=3):
        pde = ArctanData()

        node = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float_)
        cell = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int_)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n=n)

        space = LagrangeFiniteElementSpace(mesh, p=p)
        uh = space.function()

        A = space.stiff_matrix()
        b = space.source_vector(pde.source)

        bc = DirichletBC(space, pde.dirichlet)
        A, b = bc.apply(A, b, uh)
        uh[:] = spsolve(A, b).reshape(-1)

        error0 = space.integralalg.L2_error(pde.solution, uh)
        error1 = space.integralalg.L2_error(pde.gradient, uh.grad_value)
        print(error0)
        print(error1)

test = PdeTest()
test.Arctan(n = int(sys.argv[1]))
