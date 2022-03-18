#!/usr/bin/python3

import argparse 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF

from fealpy.ti import TetrahedronMesh
from fealpy.solver import LevelSetFEMFastSolver

import taichi as ti
import math 

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        线性有限元方法求解水平集演化方程，时间离散CN格式.
        """)

parser.add_argument('--ns',
        default=32, type=int,
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
        help='迭代多少步输出一次结果, 默认 10')
        


args = parser.parse_args()

nt = args.nt
ns = args.ns
T = args.T
step = args.step
output = args.output

ti.init()

pi = math.pi

@ti.func
def velocity(x: ti.f64, y: ti.f64, z: ti.f64) -> ti.f64:
    u0 = ti.sin(pi*x)**2*ti.sin(2*pi*y)
    u1 =-ti.sin(pi*y)**2*ti.sin(2*pi*x)
    u2 = 0.0
    return  u0, u1, u2

@ti.func
def sphere(x: ti.f64, y: ti.f64, z: ti.f64) -> ti.f64:
    val = ti.sqrt((x - 0.5)**2 + (y - 0.75)**2 + (z - 0.5)**2) - 0.15
    return val


# 生成初始网格
domain = [0, 1, 0, 1, 0, 1]
node, cell = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet', returnnc=True)

# 构造网格数据结构
mesh = TetrahedronMesh(node, cell)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 插值出恒定速度场
u = ti.field(ti.f64, (NN, 3))
mesh.linear_vector_interpolation(velocity, u)

# 插值符号距离函数
phi0 = ti.field(ti.f64, NN)
mesh.linear_scalar_interpolation(sphere, phi0)

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
    
    phi = solver.solve(b,tol=1e-12)
    phi0.from_numpy(phi)

    if (output != 'None') and ((i+1)%step == 0):
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.to_vtk(fname, nodedata = {'phi':phi0.to_numpy(), 'velocity':u.to_numpy()})
    
    # 时间步进一层 
    timeline.advance()
