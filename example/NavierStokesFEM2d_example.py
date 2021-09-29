
import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        IPCS 方法求解 Navier-Stokes 方程 
        """)

parser.add_argument('--degree',
        default=2, type=int,
        help='速度 Lagrange 有限元空间的次数, 默认为 2 次， 压力空间的次数比速度空间低 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--ns',
        default=10, type=int,
        help='空间各个方向剖分段数， 默认剖分 10 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')


args = parser.parse_args()

degree = args.degree
dim = args.dim
ns = args.ns
nt = args.nt

eps = 1e-12
T = 10
rho = 1
mu = 1
inp = 8.0
outp = 0.0

def is_in_flow_boundary(p):
    return np.abs(p[..., 0]) < eps 

def is_out_flow_boundary(p):
    return np.abs(p[..., 0] - 1.0) < eps

def is_wall_boundary(p):
    return (np.abs(p[..., 1]) < eps) || (np.abs(p[..., 1] - 1.0) < eps)




domain = [0, 1, 0, 1]

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

uspace = LagrangeFiniteElementSpace(smesh, p=degree)
pspace = LagrangeFiniteElementSpace(smesh, p=degree)

u0 = uspace.function(dim=2)
u1 = uspace.function(dim=2)

p0 = pspace.function()
p1 = psapce.function()

for i in range(0, nt): 
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    uh0[:] = uh1
    uh1[:] = 0.0

    # 时间步进一层 
    tmesh.advance()
