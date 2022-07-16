import numpy as np
import matplotlib.pylab as plt
import argparse

from fealpy.mesh import StructureHexMesh


parser = argparse.ArgumentParser(description=
                                 """
                                 在三维Yee网格上用有限差分(FDTD)求解Maxwell方程,
                                 并且没有ABC(吸收边界条件)
                                 """)

parser.add_argument('--ND',
                    default=20, type=int,
                    help='每个波长剖分的网格数，默认 20 个网格.')

parser.add_argument('--R',
                    default=0.5, type=int,
                    help='网比（时间稳定因子）， 默认为 0.5，三维情形要小于等于 1/sqrt(3)')

parser.add_argument('--NS',
                    default=20, type=int,
                    help='每个方向的网格数，默认为 20')

parser.add_argument('--NT',
                    default=150, type=int,
                    help='时间步，默认为 150')

args = parser.parse_args()

ND = args.ND
R = args.R
NT = args.NT
NS = args.NS

mesh = StructureHexMesh([0, NS, 0, NS, 0, NS], nx=NS, ny=NS, nz=NS)

Ex, Ey, Ez = mesh.function(etype='edge') # 网格边上的离散函数
Hx, Hy, Hz = mesh.function(etype='face') # 网格面上的离散函数


for n in range(NT):
    # HX: (nx+1, ny, nz) 
    # EY: (nx+1, ny, nz+1) 
    # EZ: (nx+1, ny+1, nz)
    Hx[1:-1, :, :] += R*(np.diff(Ey[1:-1, :, :], axis=2) - np.diff(Ez[1:-1, :, :], axis=1))

    # HY: (nx, ny+1, nz) 
    # EZ: (nx+1, ny+1, nz) 
    # EX: (nx, ny+1, nz+1)
    Hy[:, 1:-1, :] += R*(np.diff(Ez[:, 1:-1, :], axis=0) - np.diff(Ex[:, 1:-1, :], axis=2))

    # HZ: (nx, ny, nz+1) 
    # EX: (nx, ny+1, nz+1) 
    # EY: (nx+1, ny, nz+1)
    Hz[:, :, 1:-1] += R*(np.diff(Ex[:, :, 1:-1], axis=1) - np.diff(Ey[:, :, 1:-1], axis=0))

    # EX: (nx, ny+1, nz+1)
    # HZ: (nx, ny, nz+1)
    # HY: (nx, ny+1, nz)
    Ex[:, 1:-1, 1:-1] += R*(np.diff(Hz[:, :, 1:-1], axis=1) - np.diff(Hy[:, 1:-1, :], axis=2))

    # EY: (nx+1, ny, nz+1), 
    # HX: (nx+1, ny, nz)
    # HZ: (nx, ny, nz+1)
    Ey[1:-1, :, 1:-1] += R*(np.diff(Hx[1:-1, :, :], axis=2) - np.diff(Hz[:, :, 1:-1], axis=0))

    # Ez: (nx+1, ny+1, nz)
    # Hy: (nx, ny+1, nz)
    # Hx: (nx+1, ny, nz)
    Ez[1:-1, 1:-1, :] += R*(np.diff(Hy[:, 1:-1, :], axis=0) - np.diff(Hx[1:-1, :, :], axis=1))

    Ez[NS//2, NS//2, NS//2] += np.sin(2*n*np.pi*(R/ND))

plt.title("t=100")
plt.imshow(Ez[..., -1], cmap='jet', extent=[0, NS, 0, NS])
plt.colorbar()
plt.show()
