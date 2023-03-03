#!/usr/bin/env python3
#

import argparse
import numpy as np
from fealpy.mesh import UniformMesh3d
from fealpy.timeintegratoralg import UniformTimeLine
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0

parser = argparse.ArgumentParser(description=
                                 """
                                 在三维网格上用有限差分求解带 PML 层的 Maxwell 方程 
                                 """)

parser.add_argument('--NS',
                    default=100, type=int,
                    help='区域 x、y 和 z 方向的剖分段数， 默认为 100 段.')

parser.add_argument('--wave_type',
                    default='plane_wave', type=str,
                    help='波的类型')

parser.add_argument('--p',
                    default=(0.8, 0.3, 0.5), type=float, nargs=2,
                    help='激励位置，默认是 (0.8, 0.3, 0.5).')

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
                    default=20, type=int,
                    help='一个波长剖分的网格段数， 默认为 20 段.')

parser.add_argument('--R',
                    default=0.3, type=int,
                    help='网比， 默认为 0.3.')

parser.add_argument('--m',
                    default=6, type=float,
                    help='')

parser.add_argument('--sigma',
                    default=100, type=float,
                    help='最大电导率，默认取 100.')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')
        
parser.add_argument('--step',
        default=10, type=int,
        help='')

args = parser.parse_args()

NS = args.NS
NP = args.NP
NT = args.NT
ND = args.ND
R = args.R
m = args.m
sigma = args.sigma
wave_type = args.wave_type

output = args.output
step = args.step

h = 1 / NS
delta = h * NP
domain = [0, 1, 0, 1, 0, 1]  # 原来的区域

def sigma_x(p):
    x = p[..., 0]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = x < 0
    val[flag] = sigma * ((0 - x[flag]) / delta) ** m
    flag = x > 1
    val[flag] = sigma * ((x[flag] - 1) / delta) ** m
    return val

def sigma_y(p):
    y = p[..., 1]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = y < 0
    val[flag] = sigma * ((0 - y[flag]) / delta) ** m
    flag = y > 1
    val[flag] = sigma * ((y[flag] - 1) / delta) ** m
    return val

def sigma_z(p):
    z = p[..., 2]
    shape = p.shape[:-1]
    val = np.zeros(shape, dtype=np.float64)
    flag = z < 0
    val[flag] = sigma * ((0 - z[flag]) / delta) ** m
    flag = z > 1
    val[flag] = sigma * ((z[flag] - 1) / delta) ** m
    return val

domain = [0 - delta, 1 + delta, 0 - delta, 1 + delta, 0 - delta, 1 + delta]  # 增加了 PML 层的区域
mesh = UniformMesh3d((0, NS+2*NP, 0, NS+2*NP, 0, NS+2*NP), h=(h, h, h), origin=(-delta, -delta, -delta))  # 建立结构网格对象

sx0 = mesh.interpolation(sigma_x, intertype='facex')
sy0 = mesh.interpolation(sigma_y, intertype='facex')
sz0 = mesh.interpolation(sigma_z, intertype='facex')

sx1 = mesh.interpolation(sigma_x, intertype='facey')
sy1 = mesh.interpolation(sigma_y, intertype='facey')
sz1 = mesh.interpolation(sigma_z, intertype='facey')

sx2 = mesh.interpolation(sigma_x, intertype='facez')
sy2 = mesh.interpolation(sigma_y, intertype='facez')
sz2 = mesh.interpolation(sigma_z, intertype='facez')

sx3 = mesh.interpolation(sigma_x, intertype='edgex')
sy3 = mesh.interpolation(sigma_y, intertype='edgex')
sz3 = mesh.interpolation(sigma_z, intertype='edgex')

sx4 = mesh.interpolation(sigma_x, intertype='edgey')
sy4 = mesh.interpolation(sigma_y, intertype='edgey')
sz4 = mesh.interpolation(sigma_z, intertype='edgey')

sx5 = mesh.interpolation(sigma_x, intertype='edgez')
sy5 = mesh.interpolation(sigma_y, intertype='edgez')
sz5 = mesh.interpolation(sigma_z, intertype='edgez')

Bx0, By0, Bz0 = mesh.function(etype='face', dtype=np.float64)
Bx1, By1, Bz1 = mesh.function(etype='face', dtype=np.float64)
Hx0, Hy0, Hz0 = mesh.function(etype='face', dtype=np.float64)
Hx1, Hy1, Hz1 = mesh.function(etype='face', dtype=np.float64)
Dx0, Dy0, Dz0 = mesh.function(etype='edge', dtype=np.float64)
Dx1, Dy1, Dz1 = mesh.function(etype='edge', dtype=np.float64)
Ex0, Ey0, Ez0 = mesh.function(etype='edge', dtype=np.float64)
Ex1, Ey1, Ez1 = mesh.function(etype='edge', dtype=np.float64)

c1 = (2 - sz0 * R * h) / (2 + sz0 * R * h)
c2 = 2 * R / (2 + sz0 * R * h)

c3 = (2 - sx1 * R * h) / (2 + sx1 * R * h)
c4 = 2 * R / (2 + sx1 * R * h)

c5 = (2 - sy2 * R * h) / (2 + sy2 * R * h)
c6 = 2 * R / (2 + sy2 * R * h)

c7 = (2 - sy0 * R * h) / (2 + sy0 * R * h)
c8 = (2 + sx0 * R * h) / (2 + sy0 * R * h)
c9 = (2 - sx0 * R * h) / (2 + sy0 * R * h)

c10 = (2 - sz1 * R * h) / (2 + sz1 * R * h)
c11 = (2 + sy1 * R * h) / (2 + sz1 * R * h)
c12 = (2 - sy1 * R * h) / (2 + sz1 * R * h)

c13 = (2 - sx2 * R * h) / (2 + sx2 * R * h)
c14 = (2 + sz2 * R * h) / (2 + sx2 * R * h)
c15 = (2 - sz2 * R * h) / (2 + sx2 * R * h)

c16 = (2 - sz3[:, 1:-1, 1:-1] * R * h) / (2 + sz3[:, 1:-1, 1:-1] * R * h)
c17 = 2 * R / (2 + sz3[:, 1:-1, 1:-1] * R * h)

c18 = (2 - sx4[1:-1, :, 1:-1] * R * h) / (2 + sx4[1:-1, :, 1:-1] * R * h)
c19 = 2 * R / (2 + sx4[1:-1, :, 1:-1] * R * h)

c20 = (2 - sy5[1:-1, 1:-1, :] * R * h) / (2 + sy5[1:-1, 1:-1, :] * R * h)
c21 = 2 * R / (2 + sy5[1:-1, 1:-1, :] * R * h)

c22 = (2 - sy3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)
c23 = (2 + sx3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)
c24 = (2 - sx3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)

c25 = (2 - sz4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)
c26 = (2 + sy4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)
c27 = (2 - sy4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)

c28 = (2 - sx5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)
c29 = (2 + sz5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)
c30 = (2 - sz5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)

if wave_type == 'point_wave':
    p = np.array(args.p)
    i, j, k = mesh.cell_location(p)

elif wave_type == 'plane_wave':
    xx = args.plane_wave_x
    is_source = lambda p: (p[..., 0] > xx) & (p[..., 0] < xx + 0.01) & (p[..., 1] > 0) & (p[..., 1] < 1) \
                         & (p[..., 2] > 0) & (p[..., 2] < 0.01)
    bc = mesh.entity_barycenter('cell')
    flag = is_source(bc)
    i, j, k = mesh.cell_location(bc[flag])

for n in range(NT):
    print('dt={}'.format(n))
    Bx1 = c1 * Bx0 - c2 * (Ez0[:, 1:, :] - Ez0[:, 0:-1, :] - Ey0[:, :, 1:] + Ey0[:, :, 0:-1])
    By1 = c3 * By0 - c4 * (Ex0[:, :, 1:] - Ex0[:, :, 0:-1] - Ez0[1:, :, :] + Ez0[0:-1, :, :])
    Bz1 = c5 * Bz0 - c6 * (Ey0[1:, :, :] - Ey0[0:-1, :, :] - Ex0[:, 1:, :] + Ex0[:, 0:-1, :])

    Hx1 = c7 * Hx0 + c8 * Bx1 - c9 * Bx0
    Hy1 = c10 * Hy0 + c11 * By1 - c12 * By0
    Hz1 = c13 * Hz0 + c14 * Bz1 - c15 * Bz0

    Dx1[:, 1:-1, 1:-1] = c16 * Dx0[:, 1:-1, 1:-1] + c17 * (Hz1[:, 1:, 1:-1] - Hz1[:, 0:-1, 1:-1] - Hy1[:, 1:-1, 1:] + Hy1[:, 1:-1, 0:-1])
    Dy1[1:-1, :, 1:-1] = c18 * Dy0[1:-1, :, 1:-1] + c19 * (Hx1[1:-1, :, 1:] - Hx1[1:-1, :, 0:-1] - Hz1[1:, :, 1:-1] + Hz1[0:-1, :, 1:-1])
    Dz1[1:-1, 1:-1, :] = c20 * Dz0[1:-1, 1:-1, :] + c21 * (Hy1[1:, 1:-1, :] - Hy1[0:-1, 1:-1, :] - Hx1[1:-1, 1:, :] + Hx1[1:-1, 0:-1, :])

    Ex1[:, 1:-1, 1:-1] = c22 * Ex0[:, 1:-1, 1:-1] + c23 * Dx1[:, 1:-1, 1:-1] - c24 * Dx0[:, 1:-1, 1:-1]
    Ey1[1:-1, :, 1:-1] = c25 * Ey0[1:-1, :, 1:-1] + c26 * Dy1[1:-1, :, 1:-1] - c27 * Dy0[1:-1, :, 1:-1]
    Ez1[1:-1, 1:-1, :] = c28 * Ez0[1:-1, 1:-1, :] + c29 * Dz1[1:-1, 1:-1, :] - c30 * Dz0[1:-1, 1:-1, :]

    Ez1[i, j, k] = 0.1 * np.sin(2 * np.pi * n * (R / ND))

    Bx0[:] = Bx1
    By0[:] = By1
    Bz0[:] = Bz1
    Hx0[:] = Hx1
    Hy0[:] = Hy1
    Hz0[:] = Hz1
    Dx0[:] = Dx1
    Dy0[:] = Dy1
    Dz0[:] = Dz1
    Ex0[:] = Ex1
    Ey0[:] = Ey1
    Ez0[:] = Ez1

    if (step != 0) and (n%step == 0):
        eps = np.sqrt(epsilon_0)
        fname = output + 'test_'+ str(n).zfill(10)
        Ex, Ey, Ez = mesh.data_edge_to_cell(Ex0, Ey0, Ez0)
        Em = np.sqrt(Ex**2 + Ey**2 + Ez**2)

        celldata = {}
        celldata['Ex'] = np.ascontiguousarray(Ex/eps) 
        celldata['Ey'] = np.ascontiguousarray(Ey/eps) 
        celldata['Ez'] = np.ascontiguousarray(Ez/eps) 
        celldata['Em'] = np.ascontiguousarray(Em/eps)
        mesh.to_vtk_file(fname, celldata=celldata)

