import numpy as np
import matplotlib.pylab as plt
import argparse

from fealpy.mesh import StructureHexMesh

class EMWaveData3D:
    def __init__(self, kappa=None):
        self.kappa = 20  # 波数
        self.s = 0.5  # 网比(时间稳定因子)

    def domian(self):
        return [0, 100, 0, 100, 0, 100]

    def dirichlet(self, n):
        return np.sin(2 * np.pi * n * (self.s / self.kappa))


parser = argparse.ArgumentParser(description=
                                 """
                                 在三维Yee网格上用有限差分(FDTD)求解Maxwell方程,
                                 并且没有ABC(吸收边界条件)
                                 """)

parser.add_argument('--NX',
                    default=20, type=int,
                    help='x轴剖分段数， 默认为 20 段.')

parser.add_argument('--NY',
                    default=20, type=int,
                    help='y轴剖分段数， 默认为 20 段.')

parser.add_argument('--NZ',
                    default=20, type=int,
                    help='Z轴剖分段数， 默认为 20 段.')

parser.add_argument('--NT',
                    default=100, type=int,
                    help='时间剖分段数， 默认为 100 段.')


args = parser.parse_args()
nx = args.NX
ny = args.NY
nz = args.NZ

nt = args.NT

pde = EMWaveData3D()
mesh = StructureHexMesh(pde.domian(), nx=nx, ny=ny, nz=nz)

EX, EY, EZ = mesh.function(etype='edge')
HX, HY, HZ = mesh.function(etype='face')


for n in range(nt):
    # HX: (nx+1, ny, nz) EZ: (nx+1, ny+1, nz), EY: (nx+1, ny, nz+1) 
    HX[:] -= pde.s * (np.diff(EZ[1:-1, :, :], axis=1) - np.diff(EY[1:-1, :, :], axis=2))

    # HY: (nx, ny+1, nz) EX: (nx, ny+1, nz+1), EZ: (nx+1, ny+1, nz) 
    HY[:] -= pde.s * (np.diff(EX[:, 1:-1, :], axis=2) - np.diff(EZ[:, 1:-1, :], axis=0))

    # HZ: (nx, ny, nz+1) EY: (nx+1, ny, nz+1), EX: (nx, ny+1, nz+1) 
    HZ[:] -= pde.s * (np.diff(EY[:, :, 1:-1], axis=0) - np.diff(EX[:, :, 1:-1], axis=1))

    # EX: (nx, ny+1, nz+1), HZ: (nx, ny, nz+1), HY: (nx, ny+1, nz)
    EX[:, 1:-1, 1:-1] += pde.s * (np.diff(HZ, axis=1) - np.diff(HY, axis=2))

    # EY: (nx+1, ny, nz+1), HX: (nx+1, ny, nz), HZ: (nx, ny, nz+1)
    EY[1:-1, :, 1:-1] += pde.s * (np.diff(HX, axis=2) - np.diff(HZ, axis=0))

    # EZ: (nx+1, ny+1, nz), HY: (nx, ny+1, nz), HX: (nx+1, ny, nz)
    EZ[1:-1, 1:-1, :] += pde.s * (np.diff(HY, axis=0) - np.diff(HX, axis=1))

    ez[int(2*nx + 2), int(2*ny + 2), int(2*nz + 2)] += pde.dirichlet(n)

plt.title("dt=100")
plt.imshow(ez[..., -1], cmap='jet', extent=pde.domian()[0:4])
plt.colorbar()
plt.show()
