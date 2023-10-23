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
from torch.nn import init
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import UniformPoUSpace, PoUA, Cos, RandomFeatureSpace
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

EXTC = 100
HC = 1.0/EXTC

start_time = time()

def factory(i: int):
    sp = RandomFeatureSpace(2, Jn, Cos(), bound=(1, PI))
    init.normal_(sp.frequency, 0.0, 0.5)
    return sp

mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
space = UniformPoUSpace(factory, mesh, 'node', pou=PoUA(), print_status=True)

mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))
del _bd_node, mesh_col
col_sub = space.collocate_sub_edge(20)


b_tensor = torch.cat([source(col_in),
                      boundary(col_bd),
                      zeros(col_sub, expand=3)], dim=0)


laplace_phi = space.laplace_basis(col_in)
del col_in
phi = space.basis(col_bd)
del col_bd
c0: Tensor = space.continue_matrix_0(col_sub)
c1: Tensor = space.continue_matrix_1(col_sub)

A_tensor = torch.cat([-laplace_phi,
                      phi,
                      c0,
                      c1], dim=0)
ratio = 100.0/A_tensor.max(dim=-1, keepdim=True)[0]
A_tensor *= ratio
b_tensor *= ratio
del laplace_phi, phi, c0, c1

A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

um = space.function(torch.from_numpy(spsolve(A.T@A, A.T@b)))
del A, b, A_tensor, b_tensor
end_time = time()

mesh_err = TriangleMesh.from_box([0, 1.0, 0, 1.0], nx=10, ny=10)
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
