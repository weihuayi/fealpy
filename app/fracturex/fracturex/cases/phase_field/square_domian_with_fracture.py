import numpy as np
import argparse

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh


from app.fracturex.fracturex.phasefield.main_solver import MainSolver
from fealpy.utils import timer
import time
import matplotlib.pyplot as plt

class square_with_circular_notch():
    def __init__(self):
        """
        @brief 初始化模型参数
        """
        E = 210
        nu = 0.3
        Gc = 2.7e-3
        l0 = 0.0133
        self.params = {'E': E, 'nu': nu, 'Gc': Gc, 'l0': l0}


    def init_mesh(self, n=3):
        """
        @brief 生成实始网格
        """
        node = bm.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=bm.float64)

        cell = bm.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=bm.int32)
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
            5.9e-3, 901)[1:]))

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

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        脆性断裂任意次自适应有限元
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='有限元计算后端, 默认为 numpy.')

parser.add_argument('--method',
        default='HybridModel', type=str,
        help='有限元方法, 默认为 HybridModel.')

parser.add_argument('--enable_adaptive',
        default=True, type=bool,
        help='是否启用自适应加密, 默认为 True.')

parser.add_argument('--marking_strategy',
        default='recovery', type=str,
        help='标记策略, 默认为重构型后验误差估计.')

parser.add_argument('--refine_method',
        default='bisect', type=str,
        help='网格加密方法, 默认为 bisect.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格加密次数, 默认为 4.')

parser.add_argument('--vtkname',
        default='test', type=str,
        help='vtk 文件名, 默认为 test.')

args = parser.parse_args()
p= args.degree
backend = args.backend
method = args.method
enable_adaptive = args.enable_adaptive
marking_strategy = args.marking_strategy
refine_method = args.refine_method
n = args.n
vtkname = args.vtkname


tmr = timer()
next(tmr)
start = time.time()
bm.set_backend(backend)
model = square_with_circular_notch()

mesh = model.init_mesh(n=n)
fname = 'square_with_a_notch_init.vtu'
mesh.to_vtk(fname=fname)

ms = MainSolver(mesh=mesh, material_params=model.params, p=p, method=method)
tmr.send('init')
if enable_adaptive:
    ms.set_adaptive_refinement(marking_strategy=marking_strategy, refine_method=refine_method)

ms.add_boundary_condition('force', 'Dirichlet', model.is_force_boundary, model.is_force(), 'y')
ms.add_boundary_condition('displacement', 'Dirichlet', model.is_dirchlet_boundary, 0)
ms.solve(vtkname=vtkname)

force = ms.Rforce
disp = ms.force_value
with open('results_model1_ada.txt', 'w') as file:
    file.write(f'force: {force}\n')
fig, axs = plt.subplots()
plt.plot(disp, force, label='Force')
plt.xlabel('Displacement Increment')
plt.ylabel('Residual Force')
plt.title('Changes in Residual Force')
plt.grid(True)
plt.legend()
plt.savefig('model1_force.png', dpi=300)

tmr.send('stop')
end = time.time()
print(f"Time: {end - start}")