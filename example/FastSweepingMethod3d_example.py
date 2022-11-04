#!/usr/bin/python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh3d 
from fealpy.geometry import HeartSurface

parser = argparse.ArgumentParser(description=
        """
        在二维结构网格上用 Fast Sweeping Method 求解符号距离函数
        """)


parser.add_argument('--NS',
        default=200, type=int,
        help='区域 x 和 y 方向的剖分段数， 默认为 200 段.')

parser.add_argument('--M',
        default= 2, type=int,
        help='填充的默认最大值， 默认为 2.')

parser.add_argument('--curve',
        default='fold', type=str,
        help='隐式曲线， 默认为 fold .')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

args = parser.parse_args()
ns = args.NS
curve = args.curve
m = args.M
output = args.output

if curve == 'fold': 
    curve  = FoldCurve(6)
else:
    pass


box = curve.box # 这里假定 box 是一个正方形区域
origin = (box[0], box[2])
extent = [0, ns, 0, ns]
h = (box[1] - box[0])/ns # 两个方向的步长一致
mesh = UniformMesh2d(extent, h=(h, h), origin=origin) # 建立结构网格对象

# 注意这里的网格函数比实际的网格函数多了一圈，可以方便后续程序实现
phi = mesh.function(ex=1) 
isNearNode = mesh.function(dtype=np.bool_, ex=1)

# 把水平集函数转化为离散的网格函数
node = mesh.entity('node')
phi[1:-1, 1:-1] = curve(node)
sign = np.sign(phi[1:-1, 1:-1])

# 标记界面附近的点
isNearNode[1:-1, 1:-1] = np.abs(phi[1:-1, 1:-1]) < 2*h

_, d = curve.project(node[isNearNode[1:-1, 1:-1]])
phi[isNearNode] = np.abs(d) #界面附近的点用精确值
phi[~isNearNode] = m  # 其它点用一个比较大的值

