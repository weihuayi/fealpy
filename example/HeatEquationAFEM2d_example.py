#!/usr/bin/env python3
# 

import argparse


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# 装饰子：指明被装饰函数输入的是笛卡尔坐标点
from fealpy.decorator import cartesian

# 网格工厂：生成常用的简单区域上的网格
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import HalfEdgeMesh2d

# 均匀剖分的时间离散
from fealpy.timeintegratoralg import UniformTimeLine

# 热传导 pde 模型
from fealpy.pde.heatequation_model_2d import ExpExpData

# Lagrange 有限元空间
from fealpy.functionspace import LagrangeFiniteElementSpace

# Dirichlet 边界条件
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve

#拷贝对象
import copy

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格自适应有限元方法求解热传导方程
        """)

parser.add_argument('--ns',
        default=10, type=int,
        help='空间各个方向剖分段数， 默认剖分 10 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--tol',
        default=0.05, type=float,
        help='自适应加密停止阈值，默认设定为 0.05.')

parser.add_argument('--rtheta',
        default=0.7, type=float,
        help='自适应加密参数，默认设定为 0.7.')

parser.add_argument('--ctheta',
        default=0.3, type=float,
        help='自适应粗化参数，默认设定为 0.3.')

args = parser.parse_args()

ns = args.ns
nt = args.nt
tol = args.tol

rtheta = args.rtheta 
ctheta = args.ctheta 

pde = ExpExpData()
domain = pde.domain()
c = pde.diffusionCoefficient

tmesh = UniformTimeLine(0, 1, nt) # 均匀时间剖分

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3) # 三角形网格的单边数据结构

smesh.add_plot(plt)
plt.savefig('./test-' + str(0) + '.png')
plt.close()
i = 0   
while True:

    # 初始网格的自适应
    space = LagrangeFiniteElementSpace(smesh, p=1) # 构造线性元空间
    # 当前时间步的有限元解
    uh0 = space.interpolation(pde.init_value)
    eta = space.recovery_estimate(uh0, method='area_harmonic')
    err = np.sqrt(np.sum(eta**2))
    if err < tol:
        break
    isMarkedCell = smesh.refine_marker(eta, rtheta, method='L2')
    smesh.refine_triangle_rg(isMarkedCell)
    i += 1
    smesh.add_plot(plt)
    plt.savefig('./test-' + str(i+1) + '.png')
    plt.close()

space = LagrangeFiniteElementSpace(smesh, p=1)
uh0 = space.interpolation(pde.init_value)

for j in range(0, nt): 

    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    while True:
        # 下一层时间步的有限元解
        uh1 = space.function()
        A = c*space.stiff_matrix() # 刚度矩阵
        M = space.mass_matrix() # 质量矩阵
        dt = tmesh.current_time_step_length() # 时间步长
        G = M + dt*A # 隐式迭代矩阵

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
        uh1[:] = spsolve(GD, F)
        eta = space.recovery_estimate(uh1, method='area_harmonic')
        err = np.sqrt(np.sum(eta**2))
        print('errrefine', err)
        if err < tol:
            break
        else:
            #加密并插值
            NN0 = smesh.number_of_nodes()
            edge = smesh.entity('edge')
            isMarkedCell = smesh.refine_marker(eta, rtheta, method='L2')
            smesh.refine_triangle_rg(isMarkedCell)
            i += 1
            smesh.add_plot(plt)
            plt.savefig('./test-'+str(i+1)+'.png')
            plt.close()
            space = LagrangeFiniteElementSpace(smesh, p=1)
            print('refinedof', space.number_of_global_dofs())
            uh00 = space.function()
            nn2e = smesh.newnode2edge
            uh00[:NN0] = uh0
            uh00[NN0:] = np.average(uh0[edge[nn2e]], axis=-1)
            uh0 = space.function()
            uh0[:] = uh00
    #粗化网格并插值
    isMarkedCell = smesh.refine_marker(eta, ctheta, 'COARSEN')
    smesh.coarsen_triangle_rg(isMarkedCell)
    i += 1
    smesh.add_plot(plt)
    plt.savefig('./test-'+str(i+1)+'.png')
    plt.close()
    space = LagrangeFiniteElementSpace(smesh, p=1)
    print('coarsendof', space.number_of_global_dofs())
    uh2 = space.function()
    retain = smesh.retainnode
    uh2[:] = uh1[retain]
    uh1 = space.function()
    uh0 = space.function()
    uh1[:] = uh2

    # t1 时间层的误差
    @cartesian
    def solution(p):
        return pde.solution(p, t1)
    error = space.integralalg.error(solution, uh1)
    print("error:", error)

    #画数值解图像
    if (t1 ==0.01) | (t1 == 0.49) | (t1==0.99):
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        uh1.add_plot(axes, cmap='rainbow')
    uh0[:] = uh1
    uh1[:] = 0.0

    # 时间步进一层 
    tmesh.advance()

plt.show()
