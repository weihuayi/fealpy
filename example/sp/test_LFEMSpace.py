


import argparse
from fealpy.symcom.LagrangeFEMSpace import LagrangeFEMSpace

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上矩阵计算
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元基函数的次数, 默认为 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='空间维数, 默认 2.')

parser.add_argument('--nrefine',
        default='mass', type=str,
        help='默认求质量矩阵.')

args = parser.parse_args()

p = args.degree
GD = args.dim

space = LagrangeFEMSpace(GD)

