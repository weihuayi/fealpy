import numpy as np

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.geometry import SquareWithCircleHoleDomain

from app.fracturex.fracturex.phasefield.main_solver import MainSolver


class square_with_circular_notch():
    def __init__(self):
        """
        @brief 初始化模型参数
        """
        E = 200
        nu = 0.2
        Gc = 1.0
        l0 = 0.1
        self.params = {'E': E, 'nu': nu, 'Gc': 1.0, 'l0': 0.1}


    def is_force(self):
        """
        @brief 位移增量条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return bm.concatenate((bm.linspace(0, 70e-3, 6), bm.linspace(70e-3,
            125e-3, 26)[1:]))

    def is_force_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = bm.abs(p[..., 1] - 1) < 1e-12 
        return isDNode

    def is_dirchlet_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return bm.abs((p[..., 0]-0.5)**2 + bm.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001


model = square_with_circular_notch()

domain = SquareWithCircleHoleDomain(hmin=0.01) 
mesh = TriangleMesh.from_domain_distmesh(domain, maxit=100)

ms = MainSolver(mesh=mesh, material_params=model.params, p=1, method='HybridModel')
        

ms.add_boundary_condition('force', 'Dirichlet', model.is_force_boundary, model.is_force, 'y')
ms.add_boundary_condition('both', 'Dirichlet', model.is_dirchlet_boundary, 0)
ms.solve(vtkname='test')