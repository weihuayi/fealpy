
import argparse 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import StructureQuadMesh

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        有限差分方法求解水平集演化方程
        """)

parser.add_argument('--ns',
        default=100, type=int,
        help='空间各个方向剖分段数， 默认剖分 100 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--T',
        default=1, type=float,
        help='演化终止时间, 默认为 1')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

nt = args.nt
ns = args.ns
T = args.T
output = args.output

def velocity_x(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    return u

def velocity_y(p):
    x = p[..., 0]
    y = p[..., 1]
    u = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val

def single_well_potential(s):
    return (s-1)**2/2.0

def double_well_potential(s):
    pi = np.pi
    val = np.zeros_like(s)
    flag = s <= 1
    val[ flag] = (1 - np.cos(2*pi*s[flag]))/(2*pi)**2
    val[~flag] = (s[~flag] - 1)**2/2.0

    return val
    



domain = [0, 1, 0, 1]
mesh = StructureQuadMesh(domain, nx=ns, ny=ns) 

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

phi0 = mesh.interpolation(circle, etype='node')
ux = mesh.interpolation(velocity_x, etype='node')
uy = mesh.interpolation(velocity_y, etype='node')

for i in range(nt):
    t1 = timeline.next_time_level()
    print("t1=", t1)


