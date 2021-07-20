
import argparse

import sys
import numpy as np
import matplotlib.pyplot as plt

from PlanetHeatConductionSimulator import PlanetHeatConductionSimulator
from TPMModel import TPMModel 

from fealpy.tools.show import showmultirate, show_error_table
from scipy.sparse.linalg import spsolve

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三棱柱网格上热传导方程任意次有限元
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--integral',
        default=6, type=int,
        help='积分精度, 默认为 6 次.')

parser.add_argument('--nrefine',
        default=0, type=int,
        help='初始网格加密的次数, 默认初始加密 0 次.')

parser.add_argument('--h',
        default=0.005, type=float,
        help='默认单个三棱柱网格的高度, 默认高度为 0.005.')

parser.add_argument('--nh',
        default=100, type=int,
        help='默认三棱柱网格的层数, 默认层数为 100.')

parser.add_argument('--scale',
        default=500, type=int,
        help='默认小行星的规模, 默认规模为 500.')

parser.add_argument('--T',
        default=10, type=int,
        help='求解的最终时间, 默认为 10 天.')

parser.add_argument('--DT',
        default=60, type=int,
        help='求解的时间步长, 默认为 60 秒.')

parser.add_argument('--accuracy',
        default=1e-10, type=float,
        help='picard 迭代的精度, 默认为 e-10.')

args = parser.parse_args()

pde = TPMModel(args)
mesh = pde.init_mesh()

simulator = PlanetHeatConductionSimulator(pde, mesh, args)

simulator.run()

uh = simulator.uh1

Tss = pde.options['Tss']
uh = uh*Tss
print('uh:', uh)

np.savetxt('01solution', uh)
mesh.nodedata['uh'] = uh

mesh.to_vtk(fname='test.vtu') 
