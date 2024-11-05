#!/usr/bin/python3
'''!    	
@Author: wpx
@File Name: level.py
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
@bref:
@ref:
'''  

import argparse 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian, barycentric
from fealpy.solver import LevelSetFEMFastSolver

import pickle
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

degree = args.degree
nt = args.nt
ns = args.ns
T = args.T
output = args.output

@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val


domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

space = LagrangeFiniteElementSpace(mesh, p=degree)
phi0 = space.interpolation(circle)
u = space.interpolation(velocity_field, dim=2)


M = space.mass_matrix()
C = space.convection_matrix(c = u).T 
A = M + dt/2*C

diff = []
measure = space.function()

if output != 'None':
    fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
    MF.write_to_vtu(fname, mesh, nodedata = {'phi':phi0, 'velocity':u},
            p=degree)

solver = LevelSetFEMFastSolver(A)

for i in range(nt):
        
    t1 = timeline.next_time_level()
    print("t1=", t1)
    
    #计算面积
    measure[phi0 > 0] = 0
    measure[phi0 <=0] = 1
    diff.append(abs(space.integralalg.integral(measure) - (np.pi)*0.15**2))

    b = M@phi0 - dt/2*(C@phi0)

    phi0[:] = solver.solve(b, tol=1e-12)

    
    if output != 'None':
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        MF.write_to_vtu(fname, mesh, nodedata = {'phi':phi0, 'velocity':u},
                p=degree)

    # 时间步进一层 
    timeline.advance()

pickle_file = open('diff'+str(ns)+'-'+str(degree)+'.pkl','wb')
pickle.dump(diff, pickle_file) # 写入文件
pickle_file.close()

plt.figure()
plt.plot(range(len(diff)), diff, '--', color='g', label='Measure Difference')
plt.legend(loc='upper right')
plt.savefig(fname = output+'measure'+'.png')
plt.show()


