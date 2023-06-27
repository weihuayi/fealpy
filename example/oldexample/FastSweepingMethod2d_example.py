#!/usr/bin/python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d, UniformMesh2dFunction
from fealpy.geometry import FoldCurve


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

curve  = FoldCurve(6)
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

lsfun = UniformMesh2dFunction(mesh, phi[1:-1, 1:-1])

#_, d = curve.project(node[isNearNode[1:-1, 1:-1]])
_, d = lsfun.project(node[isNearNode[1:-1, 1:-1]])
print(d)


phi[isNearNode] = np.abs(d) #界面附近的点用精确值
phi[~isNearNode] = m  # 其它点用一个比较大的值


a = np.zeros(ns+1, dtype=np.float64) 
b = np.zeros(ns+1, dtype=np.float64)
c = np.zeros(ns+1, dtype=np.float64)

n = 0
for i in range(1, ns+2):
    a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
    b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
    flag = np.abs(a-b) >= h 
    c[flag] = np.minimum(a[flag], b[flag]) + h 
    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])

    fname = output + 'test'+ str(n).zfill(10)
    data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
    nodedata = {'phi':data}
    mesh.to_vtk_file(fname, nodedata=nodedata)
    n += 1


for i in range(ns+1, 0, -1):
    a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
    b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
    flag = np.abs(a-b) >= h 
    c[flag] = np.minimum(a[flag], b[flag]) + h 
    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])

    fname = output + 'test'+ str(n).zfill(10)
    data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
    nodedata = {'phi':data}
    mesh.to_vtk_file(fname, nodedata=nodedata)
    n += 1

for j in range(1, ns+2):
    a[:] = np.minimum(phi[0:ns+1, j], phi[2:, j])
    b[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
    flag = np.abs(a-b) >= h 
    c[flag] = np.minimum(a[flag], b[flag]) + h 
    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    phi[1:-1, j] = np.minimum(c, phi[1:-1, j])

    fname = output + 'test'+ str(n).zfill(10)
    data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
    nodedata = {'phi':data}
    mesh.to_vtk_file(fname, nodedata=nodedata)
    n += 1

for j in range(ns+1, 0, -1):
    a[:] = np.minimum(phi[0:ns+1, j], phi[2:, j])
    b[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
    flag = np.abs(a-b) >= h 
    c[flag] = np.minimum(a[flag], b[flag]) + h 
    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    phi[1:-1, j] = np.minimum(c, phi[1:-1, j])

    fname = output + 'test'+ str(n).zfill(10)
    data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
    nodedata = {'phi':data}
    mesh.to_vtk_file(fname, nodedata=nodedata)
    n += 1


mesh.show_function(plt, phi[1:-1, 1:-1])
plt.show()

