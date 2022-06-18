#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TriMesher import distmesh 

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

parser.add_argument('--h', 
        default=0.05, type=float, 
        help="最大网格尺寸值，默认 0.05")

parser.add_argument('--animation', 
        default=True, type=bool, 
        help="最大网格尺寸值，默认 0.05")

parser.add_argument('--maxit', 
        default=500, type=int, 
        help="最大迭代次数，默认 500 次")

args = parser.parse_args()
h = args.h
domain = args.domain
maxit = args.maxit

if domain == 0:
    fd = lambda p: drectangle(p, [0.0, 1.0, 0.0, 1.0])
    fh = huniform
    bbox = [-0.01, 1.01, -0.01, 1.01]
    pfix = np.array([ (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float64)
elif domain == 1:
    fd1 = lambda p: dcircle(p, [0.2, 0.2], 0.05)
    fd2 = lambda p: drectangle(p,[0.0, 2.2, 0.0, 0.41])
    fd = lambda p: ddiff(fd2(p),fd1(p))
    def fh(p):
        h = 0.003 + 0.05*fd1(p)
        h[h>0.01] = 0.01
        return h
    bbox = [0,3,0,1]
    pfix = np.array([(0.0,0.0),(2.2,0.0),(2.2,0.41),(0.0,0.41)],dtype=np.float64)

mesh = distmesh(h, fd, fh, bbox, pfix, showanimation=True, maxit=maxit)
fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()

