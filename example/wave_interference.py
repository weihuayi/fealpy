#!/usr/bin/env python3
#
'''
Title: 波的干涉实验

Author:  梁一茹

Address: 湘潭大学  数学与计算科学学院

'''

import argparse
import numpy as np
from fealpy.mesh import UniformMesh2d
from fealpy.timeintegratoralg import UniformTimeLine 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
         """
         波的干涉实验
         """)

parser.add_argument('--NS',
                    default=100, type=int,
                    help='区域 x 和 y 方向的剖分段数， 默认为 100 段.')

parser.add_argument('--wave_type',
                    default='point_wave', type=str,
                    help='波的类型')

parser.add_argument('--p1',
                    default=(0.47, 0.47), type=float, nargs=2,
                    help='激励位置，默认是 (0.8, 0.3).')

parser.add_argument('--p2',
                    default=(0.47, 0.53), type=float, nargs=2,
                    help='激励位置，默认是 (0.8, 0.3).')

parser.add_argument('--p3',
                    default=(0.53, 0.47), type=float, nargs=2,
                    help='激励位置，默认是 (0.8, 0.3).')

parser.add_argument('--p4',
                    default=(0.53, 0.53), type=float, nargs=2,
                    help='激励位置，默认是 (0.8, 0.3).')

parser.add_argument('--plane_wave_x',
                    default=0.5, type=float,
                    help='平面波的位置')

parser.add_argument('--NP',
                    default=20, type=int,
                    help='PML 层的剖分段数， 默认为 20 段.')

parser.add_argument('--NT',
                    default=500, type=int,
                    help='时间剖分段数， 默认为 500 段.')

parser.add_argument('--ND',
                    default=10, type=int,
                    help='一个波长剖分的网格段数， 默认为 10 段.')

parser.add_argument('--R',
                    default=0.5, type=int,
                    help='网比， 默认为 0.5.')

parser.add_argument('--m',
                    default=6, type=float,
                    help='')

parser.add_argument('--sigma',
                    default=100, type=float,
                    help='最大电导率，默认取 100.')

args = parser.parse_args()

NS = args.NS
NP = args.NP
NT = args.NT
ND = args.ND
R = args.R
m = args.m
sigma = args.sigma
wave_type = args.wave_type

T0 = 0
T1 = NT
dt = 1
h = 1/NS
delta = h*NP
domain = [0, 1, 0, 1] # 原来的区域

def sigma_x(p):
    x = p[..., 0]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = x < 0
    val[flag] = sigma*((0 - x[flag])/delta)**m
    flag = x > 1
    val[flag] = sigma*((x[flag] - 1)/delta)**m
    return val

def sigma_y(p):
    y = p[..., 1]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = y < 0
    val[flag] = sigma*((0 - y[flag])/delta)**m
    flag = y > 1
    val[flag] = sigma*((y[flag] - 1)/delta)**m
    return val

domain = [0-delta, 1+delta, 0-delta, 1+delta] # 增加了 PML 层的区域
mesh = UniformMesh2d((0, NS+2*NP, 0, NS+2*NP), h=(h, h), origin=(-delta, -delta)) # 建立结构网格对象

sx1 = mesh.interpolation(sigma_x, intertype='edgex')
sy1 = mesh.interpolation(sigma_y, intertype='edgex')

sx2 = mesh.interpolation(sigma_x, intertype='edgey')
sy2 = mesh.interpolation(sigma_y, intertype='edgey')

sx3 = mesh.interpolation(sigma_x, intertype='cell')
sy3 = mesh.interpolation(sigma_y, intertype='cell')

Dx0, Dy0 = mesh.function(etype='edge', dtype=np.float64)
Dx1, Dy1 = mesh.function(etype='edge', dtype=np.float64)
Ex, Ey = mesh.function(etype='edge', dtype=np.float64)
Bz0 = mesh.function(etype='cell', dtype=np.float64)
Bz1 = mesh.function(etype='cell', dtype=np.float64)
Hz = mesh.function(etype='cell', dtype=np.float64)

c1 = (2 - sy1[:, 1:-1] * R * h) / (2 + sy1[:, 1:-1] * R * h)
c2 = 2 * R / (2 + sy1[:, 1:-1] * R * h)

c3 = (2 - sx2[1:-1, :] * R * h) / (2 + sx2[1:-1, :] * R * h)
c4 = 2 * R / (2 + sx2[1:-1, :] * R * h)

c5 = (2 + sx1[:, 1:-1] * R * h) / 2
c6 = (2 - sx1[:, 1:-1] * R * h) / 2

c7 = (2 + sy2[1:-1, :] * R * h) / 2
c8 = (2 - sy2[1:-1, :] * R * h) / 2

c9 = (2 - sx3 * R * h) / (2 + sx3 * R * h)
c10 = 2 * R / (2 + sx3 * R * h)

c11 = (2 - sy3 * R * h) / (2 + sy3 * R * h)
c12 = 2 / (2 + sy3 * R * h)

if wave_type == 'point_wave':
    p1 = np.array(args.p1)
    p2 = np.array(args.p2)
    p3 = np.array(args.p3)
    p4 = np.array(args.p4)
    i1, j1 = mesh.cell_location(p1)
    i2, j2 = mesh.cell_location(p2)
    i3, j3 = mesh.cell_location(p3)
    i4, j4 = mesh.cell_location(p4)

elif wave_type == 'plane_wave':
    xx = args.plane_wave_x
    is_source = lambda p: (p[..., 0] > xx) & (p[..., 0] < xx + 0.01) & (p[..., 1] > 0) & (p[..., 1] < 1)
    bc = mesh.entity_barycenter('cell')
    flag = is_source(bc)
    i, j = mesh.cell_location(bc[flag])

def init(axes):
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    node = mesh.entity('node')
    data = axes.pcolormesh(node[..., 0], node[..., 1], Hz, cmap='jet', vmax=0.3, vmin=-0.3)
    return data

def forward(n):
    global Hz, i1, j1, i2, j2
    t = T0 + n*dt
    if n == 0:
        return Hz, t
    else:
        Dx1[:, 1:-1] = c1 * Dx0[:, 1:-1] + c2 * (Hz[:, 1:] - Hz[:, 0:-1])


        Dy1[1:-1, :] = c3 * Dy0[1:-1, :] - c4 * (Hz[1:, :] - Hz[0:-1, :])

        Ex[:, 1:-1] += c5 * Dx1[:, 1:-1] - c6 * Dx0[:, 1:-1]
        Ey[1:-1, :] += c7 * Dy1[1:-1, :] - c8 * Dy0[1:-1, :]

        Bz1 = c9 * Bz0 + c10 * (Ex[:, 1:] - Ex[:, 0:-1] - Ey[1:, :] + Ey[0:-1, :])
        Hz *= c11
        Hz += c12 * (Bz1 - Bz0)
        Hz[i1, j1] = np.sin(2 * np.pi * n * (R / ND))
        Hz[i2, j2] = np.sin(2 * np.pi * n * (R / ND))
        Hz[i3, j3] = np.sin(2 * np.pi * n * (R / ND))
        Hz[i4, j4] = np.sin(2 * np.pi * n * (R / ND))
        Dx0[:] = Dx1
        Dy0[:] = Dy1
        Bz0[:] = Bz1

        return Hz, t

fig, axes = plt.subplots()
mesh.show_animation(fig, axes, domain, init, forward, frames=NT+1)
plt.show()
