#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TriMesher import distmesh2d
from fealpy.geometry import dcircle, drectangle
from fealpy.geometry import ddiff, huniform


parser = argparse.ArgumentParser(description=
        """
        DisMesh2d 算法生成三角形网格。
        """)

parser.add_argument('--domain', 
        default=0, type=int, 
        help=
        """
        指定要运行的例子\n
        0 ：[0, 1]^2 区域上的均匀网格\n
        1 : 带圆洞的矩形区域 [0.0, 2.2]*[0.0, 0.41]的非均匀网格\n
        """)

parser.add_argument('--hmin', 
        default=0.05, type=float, 
        help="最小网格尺寸值，默认 0.05")

parser.add_argument('--hmax', 
        default=0.1, type=float, 
        help="最大网格尺寸值，默认 0.1")

parser.add_argument('--animation', 
        default=True, type=bool, 
        help="是否显示动画，默认显示动画")

parser.add_argument('--maxit', 
        default=250, type=int, 
        help="最大迭代次数，默认 250 次")

args = parser.parse_args()
hmin = args.hmin
hmax = args.hmax
domain = args.domain
maxit = args.maxit

if domain == 0:
    fd = lambda p: drectangle(p, [0.0, 1.0, 0.0, 1.0])
    fh = huniform
    bbox = [-0.01, 1.01, -0.01, 1.01]
    pfix = np.array([ 
        (0.0, 0.0), 
        (1.0, 0.0), 
        (1.0, 1.0), 
        (0.0, 1.0)], dtype=np.float64)

elif domain == 1:
    bbox = [-1, 1, -1, 1]
    fd = lambda p: dcircle(p, [0.0, 0.0], 1.0)
    def fh(p):
        x = p[:, 0]
        y = p[:, 1]
        h = hmin + np.abs(fd(p))*0.1
        h[h>hmax] = hmax 
        return h
    pfix = None
elif domain == 2:
    fd1 = lambda p: dcircle(p, [0.2, 0.2], 0.05)
    fd2 = lambda p: drectangle(p, [0.0, 2.2, 0.0, 0.41])
    fd = lambda p: ddiff(fd2(p), fd1(p))

    def fh(p):
        h = hmin + 0.05*np.abs(fd1(p))
        h[h>hmax] =hmax 
        return h

    bbox = [0, 3, 0, 1]
    pfix = np.array([
        (0.0, 0.0), 
        (2.2, 0.0), 
        (2.2, 0.41),
        (0.0, 0.41)],dtype=np.float64)

mesh = distmesh2d(hmin, fd, fh, bbox, pfix=pfix, 
        showanimation=True, maxit=maxit)

fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

