#!/usr/bin/env python3
#

import argparse
import numpy as np
from fealpy.mesh import UniformMesh1d
from fealpy.timeintegratoralg import UniformTimeLine
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        """
        在一维网格上用有限差分求解 Maxwell 方程
        """)

parser.add_argument('--NS',
        default=200, type=int,
        help='一维区间剖分段数， 默认为 200 段.')

parser.add_argument('--NT',
        default=1000, type=int,
        help='时间剖分段数， 默认为 1000 段.')

parser.add_argument('--ND',
        default=20, type=int,
        help='一个波长剖分的网格段数， 默认为 20 段.')

parser.add_argument('--R',
        default=0.5, type=int,
        help='网比， 默认为 0.5.')

parser.add_argument('--loss',
        default=0.004, type=float,
        help='电损耗， 默认为 0.004 .')

args = parser.parse_args()

NS = args.NS
NT = args.NT
ND = args.ND
R = args.R
loss = args.loss

T0 = 0
T1 = NT

h = 1/NS

mesh = UniformMesh1d(extent=(0, NS), h=h, origin=0) # 建立结构网格对象
timeline = UniformTimeLine(T0, T1, NT)
dt = timeline.dt

H = mesh.function(etype='cell', dtype=np.float64) # 定义在网格单元上的离散函数
E = mesh.function(etype='node', dtype=np.float64) # 定义在网格节点上的离散函数

c1 = (2 - loss) / (2 + loss)
c2 = 2 / (2 + loss)

def forward(n):
    t = T0 + n*dt
    if n == 0:
        return E, t
    else:
        H[:] -= R * (E[1:] - E[0:-1])
        E[1:-1] *= c1
        E[1:-1] -= R * c2 * (H[1:] - H[0:-1])
        E[0] = np.sin(2 * np.pi * n * (R / ND))
        return E, t

box = [0, 1, -2, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, forward, frames=NT + 1)
plt.show()
