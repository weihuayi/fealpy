#!/usr/bin/python3

import argparse 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF

from fealpy.ti import TriangleMesh 
from fealpy.solver import LevelSetFEMFastSolver

import taichi as ti
import math 


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解水平集演化方程,时间离散CN格式
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--ns',
        default=100, type=int,
        help='空间各个方向剖分段数， 默认剖分 100 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--T',
        default=1, type=float,
        help='演化终止时间, 默认为 1')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')
        
parser.add_argument('--step',
        default=10, type=int,
        help='')


args = parser.parse_args()

nt = args.nt
ns = args.ns
T = args.T
output = args.output

ti.init()

pi = math.pi

@ti.func
def velocity(x: ti.f64, y: ti.f64) -> ti.f64:
    u0 = ti.sin(pi*x)**2*ti.sin(2*pi*y)
    u1 =-ti.sin(pi*y)**2*ti.sin(2*pi*x)
    return  u0, u1 

@ti.func
def circle(x: ti.f64, y: ti.f64) -> ti.f64:
    val = ti.sqrt((x - 0.5)**2 + (y - 0.75)**2) - 0.15
    return val

# 生成初始网格
domain = [0, 1, 0, 1]
node, cell = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri', returnnc=True)

# 构建网格数据结构
mesh = TriangleMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 插值出速度场
u = ti.field(ti.f64, (NN, 2))
mesh.vector_interpolation(velocity, u)

# 插值符号距离函数
phi0 = ti.field(ti.f64, NN)
mesh.scalar_interpolation(circle, phi0)

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

M = mesh.mass_matrix()
C = mesh.convection_matrix(u)
A = M + dt/2*C

if output != 'None':
    fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
    mesh.to_vtk(fname, nodedata = {'phi':phi0.to_numpy(),
        'velocity':u.to_numpy()})

solver = LevelSetFEMFastSolver(A)

for i in range(nt):
    t1 = timeline.next_time_level()
    print("t1=", t1)
    b = M@phi0.to_numpy() - dt/2*(C@phi0.to_numpy())
    phi = solver.solve(b, tol=1e-12)
    phi0.from_numpy(phi)
    
    if output != 'None':
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.to_vtk(fname, nodedata = {'phi':phi0.to_numpy(), 'velocity':u.to_numpy()})

    # 时间步进一层 
    timeline.advance()


