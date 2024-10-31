"""
The RFM for diffusion-convection-reaction 2-d problem.
Equation:
 - \\nabla\\cdot(d \\nabla u) + b\\cdot\\nabla u + c u = f in \Omega
"""

from math import sqrt
from time import time

import torch
from torch import Tensor, cos, sin
from fealpy.ml.modules import RandomFeatureSpace, Cos, Function
from fealpy.mesh import UniformMesh2d, TriangleMesh

PI = torch.pi

d = torch.tensor(0.01, dtype=torch.float64)
b = torch.tensor([10.0, 10.0], dtype=torch.float64)
c = torch.tensor(1.0, dtype=torch.float64)


Jn = 8
EXTC = 50
HC = 1/EXTC

def exact_solution(p: Tensor):
    x = p[..., 0:1]
    y = p[..., 1:2]
    return cos(PI*x) * cos(PI*y)

def boundary(p: Tensor):
    return exact_solution(p)

def source(p: Tensor):
    x = p[..., 0:1]
    y = p[..., 1:2]
    return d * PI**2 * cos(PI*x) * cos(PI*y) + d * PI**2 * cos(PI*x) * cos(PI*y)\
           - b[0] * PI * sin(PI*x) * cos(PI*y) - b[1] * PI * cos(PI*x) * sin(PI*y)\
           + c * cos(PI*x) * cos(PI*y)

start_time = time()

space = RandomFeatureSpace(2, Jn, Cos(), bound=(2*PI, PI))
space.frequency[::4, 0] = 0.0
space.frequency[1::4, 0] = PI
space.frequency[2::4, 0] = PI
space.frequency[3::4, 0] = -PI
space.frequency[::4, 1] = PI
space.frequency[1::4, 1] = 0.0
space.frequency[2::4, 1] = PI
space.frequency[3::4, 1] = PI

mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
bdry_flag = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~bdry_flag))
col_bd = torch.from_numpy(mesh_col.entity('node', index=bdry_flag))

diffusion = space.laplace_basis(col_in) * d
convection = torch.einsum("nfd, d -> nf", space.grad_basis(col_in), b)
reaction = c * space.basis(col_in)

bd_val = space.basis(col_bd)

QI = sqrt(col_in.shape[0])
QB = sqrt(col_bd.shape[0])

A_ = torch.cat([(-diffusion + convection + reaction) / QI,
                bd_val / QB], dim=0)
b_ = torch.cat([source(col_in) / QI,
                boundary(col_bd) / QB], dim=0)

um = torch.linalg.solve(A_.T@A_, A_.T@b_)
model = Function(space, um)
end_time = time()

mesh_err = TriangleMesh.from_box([0, 1, 0, 1], nx=30, ny=30)
error = model.estimate_error_tensor(exact_solution, mesh_err)
print(f"L-2 error: {error.item()}")
print(f"Time: {end_time-start_time}")

from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
qm = model.diff(exact_solution).add_pcolor(axes, [0, 1, 0, 1], [50, 50])
fig.colorbar(qm)

plt.show()
