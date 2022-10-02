#!/usr/bin/python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import StructureQuadMesh


def FSM(phi, hx):
    phi[phi>np.sqrt(2*hx**2)] = 100
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
            if np.abs(a-b)>=hx:
                c = min(a,b)+hx
            else:
                c = (a+b+np.sqrt(2*hx*hx-(a-b)**2))/2
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
            if np.abs(a-b)>=hx:
                c = min(a,b)+hx
            else:
                c = (a+b+np.sqrt(2*hx*hx-(a-b)**2))/2
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
            if np.abs(a-b)>=hx:
                c = min(a,b)+hx
            else:
                c = (a+b+np.sqrt(2*hx*hx-(a-b)**2))/2
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
            if np.abs(a-b)>=hx:
                c = min(a,b)+hx
            else:
                c = (a+b+np.sqrt(2*hx*hx-(a-b)**2))/2
            phi[i,j] = min(c,phi[i,j])
    #print('4',phi)
