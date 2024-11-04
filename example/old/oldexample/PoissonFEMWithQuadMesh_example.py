
import argparse

import numpy as np
from scipy.sparse.linalg import spsolve
import pyamg 

from fealpy.decorator import cartesian, barycentric
from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

## 参数解析

parser = argparse.ArgumentParser(description=
        """
        该示例程序采用四边形网格上的双 p(>1) 次拉格朗日有限元，
        求解区域 [0, 1]^2 上的纯 Dirichlet 边界的 Poisson 方程
        """)

parser.add_argument('--degree', 
        default=1, type=int,
        help='空间次数, 默认为 1, 即双线性元')

parser.add_argument('--nx', 
        default=10, type=int,
        help='在 x 方向上，网格的剖分段数')

parser.add_argument('--ny', 
        default=10, type=int,
        help='在 y 方向上，网格的剖分段数')

parser.add_argument('--domain', 
        action='store_const', default=[0, 1, 0, 1], const=[0, 1, 0, 1],
        help='固定求解区域 [0, 1]^2')

parser.add_argument('--output', 
        default=None, 
        help='把结果输出 vtu 格式的文件，可用 Paraview 打开, 默认为 None 不输出')

parser.print_help()
args = parser.parse_args()

domain = args.domain
degree = args.degree
nx = args.nx
ny = args.ny
output = args.output


# 开始主程序

pde = PDE()


# 创建一个双 p 次的四边形网格
mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='quad', p=degree) 

# 在 mesh 上创建一个双 p 次的参数有限元函数空间
space = ParametricLagrangeFiniteElementSpace(mesh, p=degree, spacetype='C')

# 数值解函数
uh = space.function()

# 组装刚度矩阵
A = space.stiff_matrix()

# 右端载荷
F = space.source_vector(pde.source)

# 定义边界条件
bc = DirichletBC(space, pde.dirichlet) 

# 处理边界条件
A, F = bc.apply(A, F, uh)

# 求解
uh[:] = spsolve(A, F).reshape(-1)

# 计算 L2 误差
error = space.integralalg.L2_error(pde.solution, uh)

print(error)

if output is not None:
    mesh.nodedata['uh'] = uh
    mesh.to_vtk(fname=output)

# 网格加密
# inplace = False, 表示不修改粗网格内部数据结构，
# 返回一个新的网格对象，及新网格单元与老网格单元之间的对应关系
# HB[i] 存储第 i 个新单元对应的粗网格单元
newMesh, HB = mesh.uniform_refine(n=3, inplace=False)

mesh.to_vtk(fname='coarse.vtu')
newMesh.to_vtk(fname='fine.vtu')
print(HB)
