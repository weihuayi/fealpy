#!/usr/bin/env python3
# 

import argparse
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import StructureIntervalMesh
from fealpy.timeintegratoralg import UniformTimeLine 


# 定义 PDE 模型

class WavePDE1d_0:
    def __init__(self, L, R, T0, T1, a):
        """
        @brief 构造函数
        """
        self.L = L
        self.R = R
        self.T0 = T0
        self.T1 = T1
        self.a = a

    def domain(self):
        return [self.L, self.R]

    def duration(self):
        return [self.T0, self.T1]

    def solution(self, x, t):
        """
        @brief 计算真解的值
        @note x.shape == (N, ) t.shape == (M, 1)
        """
        return 0

    def source(self, x, t):
        """
        @brief 右端项
        """
        return 0.0

    def init_solution(self, x):
        """
        @brief 初值条件
        """
        val = np.zeros_like(x)
        flag = x >= 0.7
        val[flag] = (1 - x[flag])/6.0
        val[~flag] = x[~flag]/14.0
        return val

    def diff_init_solution(self, x):
        return np.zeros_like(x) 

    def left_solution(self, t):
        return 0

    def right_solution(self, t):
        return 0



## 参数解析
parser = argparse.ArgumentParser(description=
        """
        在一维网格上用有限差分求解波动方程 
        """)

parser.add_argument('--NS',
        default=100, type=int,
        help='一维区间剖分段数， 默认为 10 段.')

parser.add_argument('--NT',
        default=1000, type=int,
        help='时间剖分段数， 默认为 100 段')

parser.add_argument('--T',
        default=5, type=float,
        help='终止时间， 默认取 5')

parser.add_argument('--theta',
        default=0.5, type=float,
        help='格式系数， 默认取 0.5')
args = parser.parse_args()

NS = args.NS
NT = args.NT
T = args.T
theta = args.theta

L = 0
R = 1
T0 = 0
T1 = T
a = 1
pde = WavePDE1d_0(L, R, T0, T1, a)

I = pde.domain()

mesh = StructureIntervalMesh(I, nx=NS)
timeline = UniformTimeLine(T0, T1, NT)

hx = mesh.hx
dt = timeline.dt

r = a*dt/hx

A0, A1, A2 = mesh.wave_equation(r, theta)

uh0 = mesh.interpolation(pde.init_solution, 'node')
duh0 = mesh.interpolation(pde.diff_init_solution, 'node')
uh1 = mesh.function('node')

uh1[0] = pde.left_solution(dt)
uh1[-1] = pde.right_solution(dt)
uh1[1:-1] = r**2*(uh0[0:-2] + uh0[2:])/2.0 + r**2*uh0[1:-1] + dt*duh0[1:-1]


def forward(n, *args):
    t = T0 + n*dt 
    if n == 0:
        return uh0, t
    elif n == 1:
        return uh1, t
    else:
        F = A1@uh1[1:-1] + A2@uh0[1:-1]
        uh0[:] = uh1[:]
        uh1[0] = pde.left_solution(t)
        uh1[-1] = pde.right_solution(t)

        F[0] += r**2*theta*uh1[0]
        F[-1] += r**2*theta*uh1[-1]

        uh1[1:-1] = spsolve(A0, F)
        return uh1, t

box = [0, 1, -1.4, 1.4]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, forward, frames=NT+1)
plt.show()
