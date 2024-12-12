from fealpy.backend import backend_manager as bm
import torch
import numpy as np
from fealpy.mesh import TetrahedronMesh, HexahedronMesh

from app.fracturex.fracturex.phasefield.main_solve import MainSolve
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
        return bm.abs(p[..., 2]) < 1e-12

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
        default=6, type=int,
        help='初始网格加密次数, 默认为 6.')

parser.add_argument('--vtkname',
        default='test', type=str,
        help='vtk 文件名, 默认为 test.')

parser.add_argument('--save_vtkfile',
        default=True, type=bool,
        help='是否保存 vtk 文件, 默认为 False.')

parser.add_argument('--force_type',
        default='y', type=str,
        help='Force type, default is y.')

parser.add_argument('--gpu', 
        default=False, type=bool,
        help='是否使用 GPU, 默认为 False.')

parser.add_argument('--cupy', 
        default=False, type=bool,
        help='是否使用cupy求解.')

args = parser.parse_args()
p= args.degree
maxit = args.maxit
backend = args.backend
model_type = args.model_type
enable_adaptive = args.enable_adaptive
marking_strategy = args.marking_strategy
refine_method = args.refine_method
n = args.n
save_vtkfile = args.save_vtkfile
vtkname = args.vtkname +'_' + args.mesh_type + '_'
force_type = args.force_type
gpu = args.gpu
cupy = args.cupy

tmr = timer()
next(tmr)
start = time.time()
bm.set_backend(backend)
if gpu:
    bm.set_default_device('cuda')
model = square_with_circular_notch_3d()

if args.mesh_type == 'hex':
    mesh = HexahedronMesh.from_crack_box()
elif args.mesh_type == 'tet':
    mesh = TetrahedronMesh.from_crack_box()
else:
    raise ValueError('Invalid mesh type.')

mesh.uniform_refine(n=n)


fname = args.mesh_type + '_3d_square_with_a_notch_init.vtu'
mesh.to_vtk(fname=fname)

ms = MainSolve(mesh=mesh, material_params=model.params, model_type=model_type)
tmr.send('init')

ms.add_boundary_condition('force', 'Dirichlet', model.is_force_boundary, model.is_z_force(), 'z')


# 固定位移边界条件
ms.add_boundary_condition('displacement', 'Dirichlet', model.is_dirchlet_boundary, 0)


if bm.backend_name == 'pytorch':
    ms.auto_assembly_matrix()
if cupy:
    ms.set_cupy_solver()

ms.output_timer()
ms.save_vtkfile(fname=vtkname)
ms.solve(p=p, maxit=maxit)

tmr.send('stop')
tmr.send(None)
end = time.time()

force = ms.get_residual_force()
disp = model.is_z_force()

ftname = 'force_'+args.mesh_type + '_p' + str(p) + '_' + 'model3d_disp.pt'

torch.save(force, ftname)
#np.savetxt('force'+tname, bm.to_numpy(force))
tname = 'params_'+args.mesh_type + '_p' + str(p) + '_' + 'model3d_disp.txt'
with open(tname, 'w') as file:
    file.write(f'\n time: {end-start},\n degree:{p},\n, backend:{backend},\n, model_type:{model_type},\n, enable_adaptive:{enable_adaptive},\n, marking_strategy:{marking_strategy},\n, refine_method:{refine_method},\n, n:{n},\n, maxit:{maxit},\n, vtkname:{vtkname}\n')
fig, axs = plt.subplots()
disp = model.is_z_force()
plt.plot(disp, force, label='Force')
plt.xlabel('Displacement Increment')
plt.ylabel('Residual Force')
plt.title('Changes in Residual Force')
plt.grid(True)
plt.legend()
pname = args.mesh_type + '_p' + str(p) + '_' + 'model3d_force.png'
plt.savefig(pname, dpi=300)

print(f"Time: {end - start}")
