import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.truss_model import Truss_3d
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm
from fealpy.fem import TrussStructureIntegrator
from fealpy.fem import DirichletBC
from scipy.sparse.linalg import spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--GD',
        default=3, type=int,
        help='模型问题的维数, 默认求解 3 维问题.')

parser.add_argument('--nrefine',
        default=2, type=int,
        help='初始网格加密的次数, 默认初始加密 2 次.')

parser.add_argument('--scale',
        default=1, type=float,
        help='网格变形系数，默认为 1')

parser.add_argument('--doforder',
        default='sdofs', type=str,
        help='自由度排序的约定，默认为 sdofs')

args = parser.parse_args()
p = args.degree
GD = args.GD
n = args.nrefine
scale = args.scale
doforder = args.doforder

# 检查参数的合法性
if GD not in [2, 3] or p < 0 or n < 0 or scale < 0 or doforder not in ['vdims', 'sdofs']:
    parser.print_help()
    exit(1)

if GD == 2:
    pde = Truss_2d()
elif GD == 3:
    pde = Truss_3d()

mesh = pde.init_mesh()

# 构建双线性型，表示问题的微分形式
space = LagrangeFESpace(mesh, p=p, doforder=doforder)
uh = space.function(dim=GD)
vspace = GD*(space, ) # 把标量空间张成向量空间
bform = BilinearForm(vspace)
E = pde.E # 杨氏模量
A0 = pde.A0 # 横截面积
bform.add_domain_integrator(TrussStructureIntegrator(E, A0))
K = bform.assembly()

 # 加载力的条件 
F = np.zeros((uh.shape[0], GD), dtype=np.float64)
idx, f = mesh.meshdata['force_bc'] # idx.shape = (2, ), f.shape = (3, )
F[idx] = f # (10, 3)

idx, disp = mesh.meshdata['disp_bc']
bc = DirichletBC(vspace, disp, threshold=idx)
A, F = bc.apply(K, F.flat, uh)

uh.flat[:] = spsolve(A, F)

print(uh)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d') 
mesh.add_plot(axes)
mesh.node += scale*uh
mesh.add_plot(axes, nodecolor='b', cellcolor='m')
plt.show()
