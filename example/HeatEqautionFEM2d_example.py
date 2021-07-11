import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.heatequation_model_2d import SinSinExpData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve



## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解热传导方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--ns',
        default=10, type=int,
        help='空间各个方向剖分段数， 默认剖分 10 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')


args = parser.parse_args()

degree = args.degree
dim = args.dim
ns = args.ns
nt = args.nt

box = [0, 1, 0, 1]
pde = SinSinExpData()
smesh = MF.boxmesh2d(box, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, 1, 100) # 均匀时间剖分

space = LagrangeFiniteElementSpace(smesh, p=degree)

A = space.stiff_matrix(c=pde.diffusionCoefficient) # 刚度矩阵
M = space.mass_matrix() # 质量矩阵
dt = tmesh.current_time_step_length() # 时间步长
G = M + dt*A # 隐式迭代矩阵

# 当前时间步的有限元解
uh0 = space.interpolation(pde.init_value)

# 下一层时间步的有限元解
uh1 = space.function()

for i in range(0, nt): 
    
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)


    # t1 时间层的右端项
    @cartesian
    def source(p):
        return pde.source(p, t1)
    F = space.source_vector(source)
    F *= dt
    F += M@uh0

    # t1 时间层的 Dirichlet 边界条件处理
    @cartesian
    def dirichlet(p):
        return pde.dirichlet(p, t1)
    bc = DirichletBC(space, dirichlet)
    GD, F = bc.apply(G, F, uh1)
    
    # 代数系统求解
    uh1[:] = spsolve(GD, F).reshape(-1)

    # t1 时间层的误差
    @cartesian
    def solution(p):
        return pde.solution(p, t1)
    error = space.integralalg.error(solution, uh1)
    print("error:", error)

    uh0[:] = uh1
    tmesh.advance()
    



