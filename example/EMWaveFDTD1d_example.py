#!/usr/bin/env python3
# 

import argparse
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import StructureIntervalMesh
from fealpy.timeintegratoralg import UniformTimeLine 

#!/usr/bin/env python3
#

import numpy as np
import matplotlib.pylab as plt
import argparse
from fealpy.mesh import StructureIntervalMesh
from fealpy.timeintegratoralg import UniformTimeLine

class EMWaveData1D:
    def __init__(self, L, R, T0, T1, loss, kappa):

        self.L = L
        self.R = R

        self.T0 = T0
        self.T1 = T1

        self.loss = loss
        self.kappa = kappa  # 波数

        self.ca = (2 - self.loss)/(2 + self.loss)
        self.cb = 2/(2 + self.loss)

    def domian(self):
        return [self.L, self.R]

    def dirichlet(self, t):
        return np.sin(2 * np.pi * t * (0.5 / self.kappa))


L = 0
R = 200

T0 = 0
T1 = 1000


parser = argparse.ArgumentParser(description=
        """
        在一维网格上用有限差分求解Maxwell方程, 这里考虑了损耗,
        并且没有ABC(吸收边界条件)
        """)

parser.add_argument('--NS',
        default=200, type=int,
        help='一维区间剖分段数， 默认为 200 段.')

parser.add_argument('--NT',
        default=100, type=int,
        help='时间剖分段数， 默认为 100 段.')

parser.add_argument('--loss',
        default=0.005, type=float,
        help='电损耗， 默认为 0.005 段.')

parser.add_argument('--kappa',
        default=20, type=float,
        help='电磁波波数， 默认为 20 段.')

args = parser.parse_args()

NS = args.NS
NT = args.NT
loss = args.loss
kappa = args.kappa

pde = EMWaveData1D(L, R, T0, T1, loss, kappa)

mesh = StructureIntervalMesh([L, R], nx=NS)
timeline = UniformTimeLine(T0, T1, NT)

dt = timeline.dt

e = mesh.function('node')
h = mesh.function('cell') 

def forward(n):
    t = T0 + n*dt
    if n == 0:
        return e, t
    else:
        h[:] = h - 0.5*np.diff(e)
        e[1:-1] = pde.ca*e[1:-1] - 0.5*pde.cb*np.diff(h)
        e[0] = pde.dirichlet(t)
        return e, t


box = [L, R, -2, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, forward, frames=NT + 1)
plt.show()

