#!/usr/bin/env python3
#

import numpy as np
import matplotlib.pylab as plt
import argparse

from fealpy.mesh import StructureQuadMesh

class EMWaveData2D:
    def __init__(self, kappa=None):
        self.kappa = 20  # 波数
        self.s = 0.5     # 网比(时间稳定因子)

    def domian(self):
        return [0, 100, 0, 100]

    def dirichlet(self, n):
        return np.sin(2 * np.pi * n * (0.5 / self.kappa))


parser = argparse.ArgumentParser(description=
        """
        在二维Yee网格上用有限差分(FDTD)求解Maxwell方程,
        并且没有ABC(吸收边界条件)
        """)

parser.add_argument('--NX',
        default=200, type=int,
        help='x轴剖分段数， 默认为 200 段.')

parser.add_argument('--NY',
        default=200, type=int,
        help='y轴剖分段数， 默认为 200 段.')

parser.add_argument('--NT',
        default=200, type=int,
        help='时间剖分段数， 默认为 200 段.')

args = parser.parse_args()
nx = args.NX
ny = args.NY
nt = args.NT

pde = EMWaveData2D()
mesh = StructureQuadMesh(pde.domian(), nx=nx, ny=ny)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

Nshape = int(np.sqrt(NN))
Cshape = int(np.sqrt(NC))

e = np.zeros([Nshape, Nshape])
hx = np.zeros([Cshape-1, Cshape])
hy = np.zeros([Cshape, Cshape-1])

for n in range(nt):
    hx[:] -= pde.s * np.diff(e[1:-1, :], axis=1)
    hy[:] += pde.s * np.diff(e[:, 1:-1], axis=0)

    e[1:-1, 1:-1] += pde.s * (np.diff(hy, axis=0) - np.diff(hx, axis=1))
    e[int(nx / 2 + 1), int(ny / 2 + 1)] += pde.dirichlet(n)


plt.imshow(e, cmap='jet', extent=pde.domian())
plt.colorbar()
plt.show()

