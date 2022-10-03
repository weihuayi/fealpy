#!/usr/bin/env python3
#

import argparse 
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix,hstack,vstack,spdiags
from scipy.sparse.linalg import spsolve

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

parser.add_argument('--DR',
        default = 0, type=int,
        help ='正则化项系数，0：无正则化项；1：single-well 2：double-well')

parser.add_argument('--mu',
        default = 0.001,type=int,\
        help='正则项前系数')



args = parser.parse_args()

degree = args.degree
nt = args.nt
ns = args.ns
T = args.T
output = args.output
mu = args.mu
DR = args.DR

@cartesian
def velocity(p):
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


@barycentric
def single_well(bcs):
    val = phi0.grad_value(bcs)
    s = np.sqrt(np.sum(val**2, axis=-1))
    s[s < 1e-10] = 1e-10
    return 1- 1/s 

@barycentric
def double_well(bcs):
    pi = np.pi
    val = phi0.grad_value(bcs)
    s = np.sqrt(np.sum(val**2, axis=-1))
    val = np.zeros_like(s)
    flag = s < 1 
    val[flag] = np.sin(2*pi*s[flag])/(2*pi)
    val[~flag] = s[~flag] - 1 
    s[s < 1e-10] = 1e-10
    val /= s
    return val

domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

space = LagrangeFiniteElementSpace(mesh, p=degree)
phi0 = space.interpolation(circle)
u = space.interpolation(velocity, dim=2)
measure = space.function()
diff = []

M = space.mass_matrix()
C = space.convection_matrix(c = u).T 
A = M + dt/2*C

solver = LevelSetFEMFastSolver(A)

for i in range(nt):
    t1 = timeline.next_time_level()
    print("t1=",t1)

    #计算面积
    measure[phi0 > 0] = 0
    measure[phi0 <=0] = 1
    diff.append(abs(space.integralalg.integral(measure) - (np.pi)*0.15**2))
    b = M@phi0 - dt/2*(C@phi0)
    if DR == 1:
        S = space.stiff_matrix(c=single_well)
        b -= mu*dt*S@phi0
    elif DR == 2:
        S = space.stiff_matrix(c=double_well)
        b -= mu*dt*S@phi0

    phi0[:] = solver.solve(b, tol=1e-12)

    
    if i%10==0:
        if output != 'None':
            fname = output + 'stest_'+ str(i+1).zfill(10) + '.vtu'
            MF.write_to_vtu(fname, mesh, nodedata = {'phi':phi0, 'velocity':u},
                            p=degree)
    
    
   
    timeline.advance()    
    
print(diff)

