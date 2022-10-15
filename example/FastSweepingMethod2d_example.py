#!/usr/bin/python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import StructureQuadMesh
from fealpy.geometry import FoldCurve
from fealpy.geometry.geoalg import project

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

args = parser.parse_args()
ns = args.NS
curve = args.curve
m = args.M

def update(val, a, b, c, h):
    flag = np.abs(a-b) >= h 
    c[flag] = np.minimum(a[flag], b[flag]) + h 
    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    val = np.minimum(c, val)
    return val


if curve == 'fold': 
    curve  = FoldCurve(6)
else:
    pass


domain = curve.box
mesh = StructureQuadMesh(domain, nx=ns, ny=ns) # 建立结构网格对象
h = mesh.hx

phi = mesh.interpolation(curve).reshape(-1)
sign = np.sign(phi)

node = mesh.entity('node')
edge = mesh.entity('edge')

NN = mesh.number_of_nodes()
isNearNode = np.zeros(NN, dtype=np.bool_)
isCutEdge = np.prod(sign[edge], axis=-1) <= 0

isNearNode[edge[isCutEdge]] = True

p, d = curve.project(node[isNearNode])
phi[isNearNode] = np.abs(d)
phi[~isNearNode] = m 

phi.reshape(ns+1, ns+1)

a = np.zeros(NS+1, dtype=np.float64) 
b = np.zeros(NS+1, dtype=np.float64)
c = np.zeros(NS+1, dtype=np.float64)

a[:] = phi[1, :]
b[0] = phi[0, 1]
b[-1] = phi[0, -2]
b[1:-1] = np.minimum(phi[0, 0:-1], phi[0, 1:])
p[0, :] = update(p[0, :], a, b, c, h)
for i in range(1, ns):
    a[:] = np.minimum(phi[i-1, :], phi[i+1, :])
    b[0] = phi[i, 1]
    b[-1] = phi[i, -2]
    b[1:-1] = np.minimum(phi[i, 0:-1], phi[0, 1:])
    p[i, :] = update(p[i, :], a, b, c, h)

a[:] = phi[-2, :]
b[0] = phi[ns, 1]
b[-1] = phi[ns, -2]
b[1:-1] = np.minimum(phi[ns, 0:-1], phi[NS, 1:])
p[ns, :] = update(p[ns, :], a, b, c, h)





mesh.show_function(plt, phi)
plt.show()

def FSM(phi, h):
    phi[phi>np.sqrt(2*h**2)] = 100
    #print('0',phi)
    for i in range(0,ns+1):
        for j in range(0,ns+1):
            if i== 0:
                a = phi[1,j]
            elif i == ns:
                a = phi[ns-1,j]
            else:
                a = min(phi[i+1,j],phi[i-1,j])
            if j == 0:
                b = phi[i,1]
            elif j == ns:
                b = phi[i,ns-1]
            else:
                b = min(phi[i,j-1],phi[i,j+1])
            #print(a,b)
            if np.abs(a-b)>=h:
                c = min(a,b)+h
            else:
                c = (a+b+np.sqrt(2*h*hx-(a-b)**2))/2
            #print(c)
            phi[i,j] = min(c,phi[i,j])
    #print('1',phi)
    for i in range(ns,-1,-1):
        for j in range(0,ns+1):
            if i == 0 :
                a = phi[1,j]
            elif i == ns:
                a = phi[ns-1,j]
            else:
                a = min(phi[i+1,j],phi[i-1,j])
            if j == 0:
                b = phi[i,1]
            elif j == ns:
                b = phi[i,ns-1]
            else:
                b = min(phi[i,j-1],phi[i,j+1])
            if np.abs(a-b)>=h:
                c = min(a,b)+h
            else:
                c = (a+b+np.sqrt(2*h*hx-(a-b)**2))/2
            phi[i,j] = min(c,phi[i,j])
    #print('2',phi)
    for i in range(ns,-1,-1):
        for j in range(ns,-1,-1):
            if i == 0 :
                a = phi[1,j]
            elif i == ns:
                a = phi[ns-1,j]
            else:
                a = min(phi[i+1,j],phi[i-1,j])
            if j == 0:
                b = phi[i,1]
            elif j == ns:
                b = phi[i,ns-1]
            else:
                b = min(phi[i,j-1],phi[i,j+1])
            if np.abs(a-b)>=h:
                c = min(a,b)+h
            else:
                c = (a+b+np.sqrt(2*h*hx-(a-b)**2))/2
            phi[i,j] = min(c,phi[i,j])
    #print('3',phi)

    for i in range(0,ns+1):
        for j in range(ns,-1,-1):
            if i ==0 :
                a = phi[1,j]
            elif i == ns:
                a = phi[ns-1,j]
            else:
                a = min(phi[i+1,j],phi[i-1,j])
            if j == 0:
                b = phi[i,1]
            elif j == ns:
                b = phi[i,ns-1]
            else:
                b = min(phi[i,j-1],phi[i,j+1])
            if np.abs(a-b)>=h:
                c = min(a,b)+h
            else:
                c = (a+b+np.sqrt(2*h*hx-(a-b)**2))/2
            phi[i,j] = min(c,phi[i,j])
    #print('4',phi)
