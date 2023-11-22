"""
RFM with PoUA for 2-d Poisson problem

Equation:
  - \\Delta u = f

Exact solution:
  \\exp^{-0.5(x^2 + y^2)}
"""

from time import time

import torch
from torch import Tensor, exp
from scipy.sparse.linalg import spsolve

from fealpy.ml.modules import PoUSpace, Cos, RandomFeatureSpace
from fealpy.ml.operators import (
    Form, ScalerDiffusion, ScalerMass,
    Continuous0, Continuous1
)
from fealpy.ml.sampler import ISampler, InterfaceSampler
from fealpy.mesh import UniformMesh2d, TriangleMesh

PI = torch.pi

def exact_solution(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return exp(-0.5 * (x**2 + y**2))

def boundary(p: Tensor):
    return exact_solution(p)

def source(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return -(x**2 + y**2 - 2) * exp(-0.5 * (x**2 + y**2))

def zeros(p: Tensor, expand: int=1):
    raw = p.reshape(-1, p.shape[-1])
    return torch.zeros((raw.shape[0]*expand, 1), dtype=p.dtype, device=p.device)


EXT = 3
H = 1.0/EXT
Jn = 100
N = 100

start_time = time()

def factory(i: int):
    sp = RandomFeatureSpace(2, Jn, Cos(), bound=(PI/2, PI))
    return sp

# 按均匀网格分区 构建 RFM 模型
mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
space = PoUSpace.from_uniform_mesh(factory, mesh, part_loc='node', print_status=True)

# 获得采样点
col_in = ISampler([0, 1, 0, 1], mode='linspace').run(N, N)
col_left = ISampler([0, 0, 0, 1], mode='linspace').run(1, N)
col_right = ISampler([1, 1, 0, 1], mode='linspace').run(1, N)
col_btm = ISampler([0, 1, 0, 0], mode='linspace').run(N, 1)
col_top = ISampler([0, 1, 1, 1], mode='linspace').run(N, 1)
col_bd = torch.cat([col_left, col_right, col_btm, col_top], dim=0)

_css = InterfaceSampler(mesh, part_loc='node', mode='linspace')
col_sub = _css.run(20, entity_type=True)
sub2part, sub2normal = _css.sub_to_part(return_normal=True)

# 组装矩阵 求解
form = Form(space)
form.add(col_in, ScalerDiffusion(), source)
form.add(col_bd, ScalerMass(), boundary)
form.add(col_sub, Continuous0(sub2part))
form.add(col_sub, Continuous1(sub2part, sub2normal))

A, b = form.assembly(rescale=100.)
um = space.function(torch.from_numpy(spsolve(A, b)))

end_time = time()

mesh_err = TriangleMesh.from_box([0., 1., 0., 1.], nx=16, ny=16)
error = um.estimate_error_tensor(exact_solution, mesh=mesh_err)
print(f"L-2 error: {error.item()}")
print(f"Time: {end_time - start_time}")


# Visualize

from matplotlib import pyplot as plt
fig = plt.figure("RFM for 2d poisson equation")

axes = fig.add_subplot(111)
qm = um.diff(exact_solution).add_pcolor(axes, box=[0, 1.0, 0, 1.0], nums=[100, 100])
axes.set_xlabel('x')
axes.set_ylabel('y')
fig.colorbar(qm)

plt.show()
