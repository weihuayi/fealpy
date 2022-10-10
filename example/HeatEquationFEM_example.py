#!/usr/bin/env python3
# 

import argparse


import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# 装饰子：指明被装饰函数输入的是笛卡尔坐标点
from fealpy.decorator import cartesian

# 网格工厂：生成常用的简单区域上的网格
from fealpy.mesh import MeshFactory as MF

# 均匀剖分的时间离散
from fealpy.timeintegratoralg import UniformTimeLine

# 热传导 pde 模型
from fealpy.pde.heatequation_model_2d import SinSinExpData, ExpCosData

# Lagrange 有限元空间
from fealpy.functionspace import LagrangeFiniteElementSpace

# Dirichlet 边界条件
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

pde = SinSinExpData(k=1/16)
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, 1, nt) # 均匀时间剖分

space = LagrangeFiniteElementSpace(smesh, p=degree)

mmesh = smesh.copy()
NN = mmesh.number_of_nodes()
cell = mmesh.entity("cell")
mmesh.node = np.c_[mmesh.node, np.zeros((NN, 1), dtype=np.float_)]
node = mmesh.node

c = pde.diffusionCoefficient
A = space.stiff_matrix() # 刚度矩阵
M = space.mass_matrix() # 质量矩阵
dt = tmesh.current_time_step_length() # 时间步长
G = M + c*dt*A # 隐式迭代矩阵

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
    GD, F = bc.apply(G, F)
    
    # 代数系统求解
    uh1[:] = spsolve(GD, F).reshape(-1)

    # t1 时间层的误差
    @cartesian
    def solution(p):
        return pde.solution(p, t1)
    error = space.integralalg.error(solution, uh1)
    print("error:", error)

    uh0[:] = uh1[:].copy()
    #uh1[:] = 0

    # 时间步进一层 
    tmesh.advance()

    node[cell[:, 0], 2] = uh0(np.array([[1, 0, 0]]))
    node[cell[:, 1], 2] = uh0(np.array([[0, 1, 0]]))
    node[cell[:, 2], 2] = uh0(np.array([[0, 0, 1]]))

    mmesh.nodedata["u"] = uh0[:]
    s = ("%i"%(i)).zfill(3)
    mmesh.to_vtk(fname="u" + s + ".vtu")




