#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.geometry import RectangleDomain, CircleDomain, LShapeDomain
from fealpy.mesh import TriangleMesh


parser = argparse.ArgumentParser(description=
        """
        DisMesher2d 算法生成三角形网格。

        指定要运行的例子\n
        0 : [0, 1]^2 区域上的均匀网格\n
        1 :  
        """)

parser.add_argument('--domain', 
        default=0, type=int, 
        help="区域类型，默认 square")

parser.add_argument('--hmin', 
        default=0.05, type=float, 
        help="最小网格尺寸值，默认 0.05")

parser.add_argument('--hmax', 
        default=0.2, type=float, 
        help="最大网格尺寸值，默认0.2")

parser.add_argument('--animation', 
        default=True, type=bool, 
        help="是否显示动画，默认显示动画")

parser.add_argument('--maxit', 
        default=250, type=int, 
        help="最大迭代次数，默认 250 次")

args = parser.parse_args()
domain = args.domain
hmin = args.hmin
hmax = args.hmax
maxit = args.maxit


if domain == 0:
    box = [0, 1, 0, 1]
    domain = RectangleDomain(box)
elif domain == 1:
    domain = LShapeDomain()
elif domain == 2:
    def sizing_function(p, *args):
        fd = args[0]
        x = p[:, 0]
        y = p[:, 1]
        h = hmin + np.abs(fd(p))*0.1
        h[h>hmax] = hmax 
        return h
    domain = CircleDomain(fh=sizing_function)

mesh = TriangleMesh.from_domain_distmesh(domain, hmin, maxit=maxit)

c = mesh.circumcenter()
fig, axes = plt.subplots()
mesh.add_plot(axes)
mesh.find_node(axes, node=c)
plt.show()

