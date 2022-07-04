#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fealpy.geometry import SphereDomain
from fealpy.mesh import DistMesher3d 


parser = argparse.ArgumentParser(description=
        """
        DisMesher3d 算法生成四面体网格。
        """)

parser.add_argument('--hmin', 
        default=0.1, type=float, 
        help="最小网格尺寸值，默认 0.1")

parser.add_argument('--hmax', 
        default=0.1, type=float, 
        help="最大网格尺寸值，默认与最小网格尺寸相同")

parser.add_argument('--animation', 
        default=True, type=bool, 
        help="是否显示动画，默认显示动画")

parser.add_argument('--maxit', 
        default=250, type=int, 
        help="最大迭代次数，默认 250 次")

args = parser.parse_args()
hmin = args.hmin
hmax = args.hmax
maxit = args.maxit

domain = SphereDomain()

mesher = DistMesher3d(domain, hmin, output=True)
mesh = mesher.meshing(maxit)

mesh.to_vtk(fname='test.vtu')

