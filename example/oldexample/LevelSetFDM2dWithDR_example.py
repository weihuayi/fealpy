#!/usr/bin/env python3
#

import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.timeintegratoralg import UniformTimeLine#剖分时间
from fealpy.mesh import StructureQuadMesh
from mpl_toolkits.mplot3d import Axes3D

##参数解析
parser = argparse.ArgumentParser(description=
        """
        有限差分方法求解水平集演化方程
        """)

parser.add_argument('--ns',
        default =100 ,type=int,
        help='空间各个方向剖分段数，默认剖分100段.')

parser.add_argument('--nt',
        default =1200,type = int,
        help = '时间剖分段数，默认剖分100段.')

parser.add_argument('--T',
        default = 1,type=float,
        help = '演化终止时间，默认为1')

parser.add_argument('--output',
        default = './',type=str,
        help ='结果输出目录，默认为 ./')

parser.add_argument('--DR',
        default = 0, type=int,
        help ='正则化项系数，0：无正则化项；1：single-well 2：double-well')

parser.add_argument('--mu',
        default = 0.001,type=float,
        help ='正则项系数，默认值为 0.001')

args = parser.parse_args()

ns = args.ns
nt = args.nt
T = args.T
output = args.output
DR  = args.DR
mu = args.mu

def velocity_x(p):
    x = p[...,0]
    y = p[...,1]
    u = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    return u

def velocity_y(p):
    x = p[...,0]
    y = p[...,1]
    u = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

def velocity(p):
    x = p[...,0]
    y = p[...,1]
    u = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    return u

def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val

def single_well(s):
    s[s < 1e-6] = 1e-6
    return 1- 1/s 

def double_well(s):
    pi = np.pi
    val = np.zeros_like(s)
    flag = s < 1 
    val[flag] = np.sin(2*pi*s[flag])/(2*pi)
    val[~flag] = s[~flag] - 1 
    s[s < 1e-6] = 1e-6
    val /= s
    return val

#求拉普拉斯算子
def del2(A):
    a = np.zeros(np.shape(A))
    a[1:-1,1:-1] = (A[0:-2,1:-1]+A[2:,1:-1]-4*A[1:-1,1:-1]+A[1:-1,:-2]+A[1:-1,2:])/4
    
    a[0,0] = A[0,0]+A[0,2]+A[2,0]-A[0,1]-A[1,0]-(A[0,1]+A[1,0]+A[0,3]+A[3,0])/4
    a[0,-1] = A[0,-1]+A[0,-3]+A[2,-1]-A[0,-2]-A[1,-1]-(A[0,-2]+A[1,-1]+A[0,-4]+A[3,-1])/4
    a[-1,0] = A[-1,0]+A[-3,0]+A[-1,2]-A[-2,0]-A[-1,1]-(A[-2,0]+A[-1,1]+A[-4,0]+A[-1,3])/4
    a[-1,-1] = A[-1,-1]+A[-1,-3]+A[-3,-1]-A[-1,-2]-A[-2,-1]-(A[-1,-2]+A[-2,-1]+A[-1,-4]+A[-4,-1])/4
    
    a[0,1:-1] = A[2,1:-1]-A[1,1:-1]+(A[0,:-2]+A[0,2:]-A[3,1:-1]-A[1,1:-1])/4
    a[1:-1,0] = A[1:-1,2]-A[1:-1,1]+(A[:-2,0]+A[2:,0]-A[1:-1,3]-A[1:-1,1])/4
    a[1:-1,-1] = A[1:-1,-3]-A[1:-1,-2]+(A[:-2,-1]+A[2:,-1]-A[1:-1,-4]-A[1:-1,-2])/4
    a[-1,1:-1] = A[-3,1:-1]-A[-2,1:-1]+(A[-1,:-2]+A[-1,2:]-A[-4,1:-1]-A[-2,1:-1])/4
    return a 

#散度
def div(fx,fy):
    fxx,fxy = np.gradient(fx)
    fyx,fyy = np.gradient(fy)
    f = fxx+fyy
    return f

domain = [0, 1, 0, 1]

mesh = StructureQuadMesh(domain, nx=ns, ny=ns)

hx = mesh.hx
hy = mesh.hy

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

phi0 = mesh.interpolation(circle,intertype='node')
ux = mesh.interpolation(velocity_x,intertype='node')
uy = mesh.interpolation(velocity_y,intertype='node')


#赋值
phi = phi0
t1 = 0
#网比
r = dt/(hx)**2
print(r,'r')

for i in range(nt):
    t1 = timeline.next_time_level()
    print("t1=",t1)
    phi_x,phi_y = np.gradient(phi)
    s = np.sqrt((phi_x/hx)**2+(phi_y/hx)**2)    
    #print('s',s)    
   
    if potentialFunction =="single-well":
        phi = phi+r*(mu*(4*del2(phi)-div(phi_x/s,phi_y/s))-hx*(ux*phi_x+uy*phi_y))

   
    elif potentialFunction =="double-well":
        dps = d_double_well_potential(s)/s
        phi = phi+r*(mu*(4*del2(phi)+div(dps*phi_x-phi_x,dps*phi_y-phi_y))-hx*(phi_x*ux+phi_y*uy))
    
    #数据传为vtu文件，画图
    if output != 'None':        
        if i%50==0:
            mesh.nodedata['phi'] = phi.flatten()
            fname = output + 'mu0.001'+ str(i+1).zfill(10) + '.vtu'
            mesh.to_vtk(etype='cell', fname=fname)
    # 时间步进一层 
    timeline.advance()
   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
