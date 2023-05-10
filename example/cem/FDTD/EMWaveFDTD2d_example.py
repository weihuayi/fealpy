#!/usr/bin/env python3
#

import argparse
import numpy as np
from fealpy.mesh import UniformMesh2d
from fealpy.timeintegratoralg import UniformTimeLine
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        """
        在二维网格上用有限差分求解 Maxwell 方程,
        """)

parser.add_argument('--NS',
        default=200, type=int,
        help='区域 x 和 y 方向的剖分段数， 默认为 200 段.')

parser.add_argument('--NT',
        default=500, type=int,
        help='时间剖分段数， 默认为 500 段.')

parser.add_argument('--ND',
        default=20, type=int,
        help='一个波长剖分的网格段数， 默认为 20 段.')

parser.add_argument('--R',
        default=0.5, type=int,
        help='网比， 默认为 0.5.')

args = parser.parse_args()

NS = args.NS
NT = args.NT
ND = args.ND
R = args.R

T0 = 0
T1 = NT

h = 100/NS

domain = [0, 100, 0, 100] # 笛卡尔坐标空间
mesh = UniformMesh2d((0, NS, 0, NS), h=(h, h), origin=(0, 0)) # 建立结构网格对象
timeline = UniformTimeLine(T0, T1, NT)
dt = timeline.dt

Hx, Hy = mesh.function(etype='edge', dtype=np.float64) # 定义在网格边上的离散函数
Ez = mesh.function(etype='cell', dtype=np.float64) # 定义在网格单元上的离散函数

i = NS // 2

def init(axes):
    data = axes.imshow(Ez, cmap='jet', vmin=-0.2, vmax=0.2, extent=domain)
    return data

def forward(n):
    global Ez
    t = T0 + n*dt
    if n == 0:
        return Ez, t
    else:
        Hx[:, 1:-1] -= R * (Ez[:, 1:] - Ez[:, 0:-1])
        Hy[1:-1, :] += R * (Ez[1:, :] - Ez[0:-1, :])
        Ez += R * (Hy[1:, :] - Hy[0:-1, :] - Hx[:, 1:] + Hx[:, 0:-1])

        Ez[i, i] = np.sin(2 * np.pi * n * (R / ND))
        return Ez, t

fig, axes = plt.subplots()
mesh.show_animation(fig, axes, domain, init, forward, frames=NT + 1)
plt.show()

