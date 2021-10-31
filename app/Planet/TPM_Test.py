
import argparse

import numpy as np

from fealpy.writer import VTKMeshWriter
#from PlanetHeatConductionSimulator_picard import PlanetHeatConductionWithRotationSimulator, PlanetHeatConductionWithIrrotationSimulator
from PlanetHeatConductionSimulator_newton import PlanetHeatConductionWithRotationSimulator
from TPMModel import TPMModel 

from mumps import DMumpsContext

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三棱柱网格上热传导方程任意次有限元
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nq',
        default=3, type=int,
        help='积分精度, 默认为 3.')

parser.add_argument('--T',
        default=20, type=float,
        help='求解的最终时间, 默认为 20 天.')

parser.add_argument('--DT',
        default=18, type=int,
        help='求解的时间步长, 默认为 18 秒.')

parser.add_argument('--accuracy',
        default=1e-10, type=float,
        help='picard 迭代的精度, 默认为 1e-10.')

parser.add_argument('--niteration',
        default=30, type=int,
        help='迭代的最大迭代次数, 默认为 30 次.')

parser.add_argument('--stable',
        default=1, type=float,
        help='默认小行星达到稳定状态时两相邻周期同一相位的最大温差为 1.')

parser.add_argument('--step',
        default=1, type=int,
        help='结果输出的步数间隔，默认为 1 步输出一次 vtu 文件.')

parser.add_argument('--output', 
        default='test', type=str,
        help='结果输出文件的主名称，默认为 test')

parser.add_argument('--nrefine',
        default=0, type=int,
        help='初始网格加密的次数, 默认初始加密 0 次.')

parser.add_argument('--h',
        default=0.5, type=float,
        help='求解的球壳厚度, 默认厚度为 0.5.')

parser.add_argument('--nh',
        default=100, type=int,
        help='默认三棱柱网格的层数, 默认层数为 100.')

parser.add_argument('--scale',
        default=500, type=int,
        help='默认小行星的规模, 默认规模为 500.')

args = parser.parse_args()

pde = TPMModel(args)
mesh = pde.test_rotation_mesh()

ctx = DMumpsContext()
ctx.set_silent()

simulator = PlanetHeatConductionWithRotationSimulator(pde, mesh, args)

writer = VTKMeshWriter(simulation=simulator.run, args=(ctx, ))
writer.run()
ctx.destroy()

