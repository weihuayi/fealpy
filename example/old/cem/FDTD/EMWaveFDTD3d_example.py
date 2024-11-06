import numpy as np
import matplotlib.pylab as plt
import argparse

from fealpy.mesh import StructureHexMesh
from fealpy.timeintegratoralg import UniformTimeLine
from scipy.constants import epsilon_0

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
                    default=200, type=int,
                    help='每个方向的网格数，默认为 200')

parser.add_argument('--NT',
                    default=500, type=int,
                    help='时间步，默认为 500')

parser.add_argument('--output',
                    default='./', type=str,
                    help='结果输出目录, 默认为 ./')

parser.add_argument('--step',
                    default=10, type=int,
                    help='')

args = parser.parse_args()

ND = args.ND
R = args.R
NT = args.NT
NS = args.NS

output = args.output
step = args.step

domain = [0, 20, 0, 20, 0, 20] # 笛卡尔坐标空间
mesh = StructureHexMesh(domain, nx=NS, ny=NS, nz=NS) # 建立结构网格对象

Hx, Hy, Hz = mesh.function(etype='face', dtype=np.float64) # 定义在网格面上的离散函数
Ex0, Ey0, Ez0 = mesh.function(etype='edge', dtype=np.float64) # 定义在网格边上的离散函数
Ex1, Ey1, Ez1 = mesh.function(etype='edge', dtype=np.float64) # 定义在网格边上的离散函数

i = NS // 2

for n in range(NT):
    print('dt={}'.format(n))
    Hx += R * (Ey0[:, :, 1:] - Ey0[:, :, 0:-1] - Ez0[:, 1:, :] + Ez0[:, 0:-1, :])
    Hy += R * (Ez0[1:, :, :] - Ez0[0:-1, :, :] - Ex0[:, :, 1:] + Ex0[:, :, 0:-1])
    Hz += R * (Ex0[:, 1:, :] - Ex0[:, 0:-1, :] - Ey0[1:, :, :] + Ey0[0:-1, :, :])

    Ex1[:, 1:-1, 1:-1] -= R * (Hy[:, 1:-1, 1:] - Hy[:, 1:-1, 0:-1] - Hz[:, 1:, 1:-1] + Hz[:, 0:-1, 1:-1])
    Ey1[1:-1, :, 1:-1] -= R * (Hz[1:, :, 1:-1] - Hz[0:-1, :, 1:-1] - Hx[1:-1, :, 1:] + Hx[1:-1, :, 0:-1])
    Ez1[1:-1, 1:-1, :] -= R * (Hx[1:-1, 1:, :] - Hx[1:-1, 0:-1, :] - Hy[1:, 1:-1, :] + Hy[0:-1, 1:-1, :])

    Ez1[i, i, i] = np.sin(2 * np.pi * n * (R / ND))

    Ex0[:] = Ex1
    Ey0[:] = Ey1
    Ez0[:] = Ez1

    if (step != 0) and (n % step == 0):
        eps = np.sqrt(epsilon_0)
        fname = output + 'test_' + str(n).zfill(10)
        Ex, Ey, Ez = mesh.data_edge_to_cell(Ex0, Ey0, Ez0)
        Em = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)

        celldata = {}
        celldata['Ex'] = np.ascontiguousarray(Ex / eps)
        celldata['Ey'] = np.ascontiguousarray(Ey / eps)
        celldata['Ez'] = np.ascontiguousarray(Ez / eps)
        celldata['Em'] = np.ascontiguousarray(Em / eps)
        mesh.to_vtk_file(fname, celldata=celldata)

