from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh, HexahedronMesh

from app.fracturex.fracturex.phasefield.main_solver import MainSolver
from fealpy.utils import timer

import time
import argparse
import matplotlib.pyplot as plt

class square_with_circular_notch_3d():
    def __init__(self):
        """
        @brief 初始化模型参数
        """
        E = 210
        nu = 0.3
        Gc = 5e-4
        l0 = 0.05
        self.params = {'E': E, 'nu': nu, 'Gc': Gc, 'l0': l0}


    def init_hex_mesh(self, n=3):
        """
        @brief 生成实始网格
        """
        pass

    def init_tet_mesh(self, n=3):
        """
        @brief 生成实始网格
        """
        pass

    def is_z_force(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return bm.linspace(0, 4e-2, 4001)

    def is_force_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = bm.abs(p[..., 2] - 10) < 1e-12
        return isDNode

    def is_dirchlet_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return np.abs(p[..., 2]) < 1e-12

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        脆性断裂任意次自适应有限元
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--maxit',
        default=30, type=int,
        help='最大迭代次数, 默认为 30 次.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='有限元计算后端, 默认为 numpy.')

parser.add_argument('--model_type',
        default='HybridModel', type=str,
        help='有限元方法, 默认为 HybridModel.')

parser.add_argument('--mesh_type',
        default='tet', type=str,
        help='网格类型, 默认为 tet.')

parser.add_argument('--enable_adaptive',
        default=False, type=bool,
        help='是否启用自适应加密, 默认为 False.')

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
maxit = args.maxit
backend = args.backend
model_type = args.model_type
enable_adaptive = args.enable_adaptive
marking_strategy = args.marking_strategy
refine_method = args.refine_method
n = args.n
vtkname = args.vtkname


tmr = timer()
next(tmr)
start = time.time()
bm.set_backend(backend)
model = square_with_circular_notch_3d()

if args.mesh_type == 'hex':
    mesh = model.init_hex_mesh(n=n)
elif args.mesh_type == 'tet':
    mesh = model.init_tet_mesh(n=n)
else:
    raise ValueError('Invalid mesh type.')

fname = 'square_with_a_notch_3d_init.vtu'
mesh.to_vtk(fname=fname)


ms = MainSolver(mesh=mesh, material_params=model.params, p=p, model_type=model_type)
tmr.send('init')

if enable_adaptive:
    print('Enable adaptive refinement.')
    ms.set_adaptive_refinement(marking_strategy=marking_strategy, refine_method=refine_method)

# 拉伸模型边界条件
ms.add_boundary_condition('force', 'Dirichlet', model.is_force_boundary, model.is_z_force(), 'z')


# 固定位移边界条件
ms.add_boundary_condition('displacement', 'Dirichlet', model.is_dirchlet_boundary, 0)

ms.solve(maxit=maxit, vtkname=vtkname)

tmr.send('stop')
end = time.time()

force = ms.Rforce
disp = ms.force_value
with open('results_model3d.txt', 'w') as file:
    file.write(f'force: {force}\n, time: {end-start}\n')
fig, axs = plt.subplots()
plt.plot(disp, force, label='Force')
plt.xlabel('Displacement Increment')
plt.ylabel('Residual Force')
plt.title('Changes in Residual Force')
plt.grid(True)
plt.legend()
plt.savefig('model3d_force.png', dpi=300)

print(f"Time: {end - start}")
