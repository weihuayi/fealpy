import numpy as np

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh


from app.fracturex.fracturex.phasefield.main_solver import MainSolver
from fealpy.utils import timer

class square_with_circular_notch():
    def __init__(self):
        """
        @brief 初始化模型参数
        """
        E = 210
        nu = 0.3
        Gc = 2.7e-3
        l0 = 0.05
        self.params = {'E': E, 'nu': nu, 'Gc': Gc, 'l0': l0}


    def init_mesh(self, n=3):
        """
        @brief 生成实始网格
        """
        node = np.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=np.float64)

        cell = np.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=np.int_)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n=n)
        return mesh

    def is_force(self):
        """
        @brief 位移增量条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return bm.concatenate((bm.linspace(0, 5e-3, 501), bm.linspace(5e-3,
            6e-3, 1001)[1:]))

    def is_force_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = bm.abs(p[..., 1] - 1) < 1e-12 
        return isDNode

    def is_dirchlet_boundary(self, p):
        """
        @brief 标记边界条件
        """
        return bm.abs(p[..., 1]) < 1e-12


tmr = timer()
next(tmr)
model = square_with_circular_notch()

mesh = model.init_mesh(n=0s)
fname = 'square_with_a_notch_init.vtu'
mesh.to_vtk(fname=fname)

ms = MainSolver(mesh=mesh, material_params=model.params, p=1, method='HybridModel')
tmr.send('init')

ms.add_boundary_condition('force', 'Dirichlet', model.is_force_boundary, model.is_force(), 'y')
ms.add_boundary_condition('displacement', 'Dirichlet', model.is_dirchlet_boundary, 0)
ms.solve(vtkname='test')
tmr.send('stop')