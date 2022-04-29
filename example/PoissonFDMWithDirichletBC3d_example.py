#!/usr/bin/env python3
# 

import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosCosData as PDE
from fealpy.mesh import StructureQuadMesh 
from fealpy.tools.show import showmultirate

from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        在三维笛卡尔网格上求解带 Dirichlet 边界的 Poisson 方程 
        """)

parser.add_argument('--nx',
        default=10, type=int,
        help='初始笛卡尔网格 x 方向剖分段数, 默认 10 段.')

parser.add_argument('--ny',
        default=10, type=int,
        help='初始笛卡尔网格 y 方向剖分段数, 默认 10 段.')

parser.add_argument('--nz',
        default=10, type=int,
        help='初始笛卡尔网格 z 方向剖分段数, 默认 10 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

args = parser.parse_args()

nx = args.nx
ny = args.ny
maxit = args.maxit
