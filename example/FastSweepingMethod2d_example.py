#!/usr/bin/python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import StructureQuadMesh
from fealpy.geometry import FoldCurve

parser = argparse.ArgumentParser(description=
        """
        在二维结构网格上用 Fast Sweeping Method 求解符号距离函数
        """)


parser.add_argument('--NS',
        default=200, type=int,
        help='区域 x 和 y 方向的剖分段数， 默认为 200 段.')

parser.add_argument('--curve',
        default='fold', type=str,
        help='隐式曲线， 默认为 fold .')

args = parser.parse_args()
NS = args.NS
curve = args.curve

if curve == 'fold': 
    curve  = FoldCurve(6)
else:
    pass


domain = curve.box
mesh = StructureQuadMesh(domain, nx=NS, ny=NS) # 建立结构网格对象

phi = mesh.interpolation(curve)
sign = np.sign(phi)

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
