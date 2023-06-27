import argparse
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.parabolic_1d import SinExpPDEData
from fealpy.mesh import UniformMesh1d

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        一维均匀网格上抛物型方程的有限差分方法，
        边界条件为纯 Dirichlet 型，
        有三种离散格式供选择：1、向前欧拉；2、向后欧拉；3、crank_nicholson。
        """)

parser.add_argument('--nx',
        default=40, type=int,
        help='空间剖分段数，默认为 40 段.')

parser.add_argument('--nt',
        default=3200, type=int,
        help='时间剖分段数，默认为 3200 段.')

parser.add_argument('--discrete_format',
        default=1, type=int,
        help=
        """
        离散格式选择：1、向前欧拉；2、向后欧拉；3、crank_nicholson，
        使用相应的数字编号选择离散格式，默认为 1、向前欧拉格式。
        """)

parser.add_argument('--box',
        default=[0, 1, -1.5, 1.5], type=list,
        help="图像显示的范围，默认为： 0 <= x <= 1, -1.5 <= y <= 1.5")

args = parser.parse_args()

nx = args.nx
nt = args.nt
discrete_format = args.discrete_format

# PDE 模型
pde = SinExpPDEData()

# 空间离散
domain = pde.domain()
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
node = mesh.node

# 时间离散
duration = pde.duration()
tau = (duration[1] - duration[0])/nt 

uh0 = mesh.interpolate(pde.init_solution, intertype='node')


def advance_forward(n):
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
    
def advance_backward(n):
    """
    @brief 时间步进格式为向后欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0
        gD = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t

def advance_crank_nicholson(n):
    """
    @brief 时间步进格式为 CN 方法  
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += B@uh0
        gD = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t


if discrete_format == 1:
    dis_format = advance_forward
elif discrete_format == 2:
    dis_format = advance_backward
elif discrete_format == 3:
    dis_format = advance_crank_nicholson
else:
    raise ValueError("请选择正确的离散格式.")

fig, axes = plt.subplots()
box = args.box
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, dis_format, frames=nt + 1)
plt.show()
