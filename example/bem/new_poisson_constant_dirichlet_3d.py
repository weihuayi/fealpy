import numpy as np
from matplotlib import pyplot as plt
import argparse

from fealpy.mesh import UniformMesh3d
from fealpy.pde.bem_model_3d import PoissonModelConstantDirichletBC3d
from fealpy.functionspace import LagrangeFESpace
from fealpy.bem import BoundaryOperator, InternalOperator, PotentialFluxIntegrator, ScalarSourceIntegrator, DirichletBC
from fealpy.bem import BoundaryElementModel
from fealpy.bem.tools import boundary_mesh_build, error_calculator
from fealpy.tools.show import showmultirate

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        UniformMesh3d 上任意次边界元方法
        """)

parser.add_argument('--degree',
        default=0, type=int,
        help='Lagrange 有限元空间的次数, 默认为 0 次.')

parser.add_argument('--nx',
        default=3, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=3, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--nz',
        default=3, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=3, type=int,
        help='默认网格加密求解的次数, 默认加密求解 3 次')

args = parser.parse_args()

pde = PoissonModelConstantDirichletBC3d()
# 定义网格对象
nx = args.nx
ny = args.ny
nz = args.nz
hx = (1 - 0) / nx
hy = (1 - 0) / ny
hz = (1 - 0) / nz
mesh = UniformMesh3d((0, nx, 0, ny, 0, nz), h=(hx, hy, hz), origin=(0, 0, 0))

p = args.degree
q = 3
maxite = args.maxit
errorMatrix = np.zeros(maxite)
N = np.zeros(maxite)

for k in range(maxite):
    # 数学模型构建与预处理
    bd_mesh = BoundaryElementModel.boundary_mesh_build(mesh)
    space = LagrangeFESpace(bd_mesh, p=p)
    space.domain_mesh = mesh

    # 边界元模型构建
    bem = BoundaryElementModel(space, PotentialFluxIntegrator(q=q-1), ScalarSourceIntegrator(f=pde.source, q=q))
    bc = DirichletBC(space=space, gD=pde.dirichlet)
    bem.boundary_condition_apply(bc)
    bem.build()

    # 内部点计算
    xi = mesh.entity('node')[~mesh.ds.boundary_node_flag()]
    u_inter = bem(xi)

    result_u = np.zeros(mesh.number_of_nodes())
    result_u[mesh.ds.boundary_node_flag()] = pde.dirichlet(mesh.entity('node')[mesh.ds.boundary_node_flag()])
    result_u[~mesh.ds.boundary_node_flag()] = u_inter

    # errorMatrix[k] = bem.error_calculate(pde.solution, q=q-1)

    errorMatrix[k] = error_calculator(mesh, result_u, pde.solution)

    v = mesh.entity_measure('cell')
    h = np.max(v)
    N[k] = np.power(h, 1 / mesh.geo_dimension())

    if k < maxite:
        mesh.uniform_refine(1)

print(f'迭代{maxite}次，结果如下：')
print("误差：\n", errorMatrix)
print('误差比：\n', errorMatrix[0:-1] / errorMatrix[1:])

fig = plt.figure()
axes = showmultirate(plt, 0, N[None, ...], errorMatrix[None, ...], labellist=[['l2']])
plt.show()