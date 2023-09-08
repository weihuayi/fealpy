"""
RFM for 2-d Poisson problem

Equation:
  - \\Delta u = f

Exact solution:
  \\exp^{-0.5(x^2 + y^2)}
"""

from time import time
from math import sqrt

import torch
from torch import Tensor, exp
from torch.nn import init
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import PoUSpace, PoUSin, Cos, Function, RandomFeatureSpace
from fealpy.mesh import UniformMesh2d, TriangleMesh

NEW_BASIS = True
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


EXT = 1
H = 1/EXT
Jn = 162

EXTC = 100
HC = 1/EXTC

start_time = time()

def factory(i: int):
    sp = RandomFeatureSpace(2, Jn, Cos(), bound=(1, PI))
    init.normal_(sp.frequency, 0.0, PI/2)
    return sp

mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
node = torch.from_numpy(mesh.entity('node')).clone()
space = PoUSpace(factory, pou=PoUSin(), centers=node, radius=H/2, print_status=True)
del mesh

mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))
del _bd_node, mesh_col

mesh_err = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)

b_tensor = torch.cat([source(col_in) / sqrt(col_in.shape[0]),
                      boundary(col_bd) / sqrt(col_bd.shape[0])], dim=0)

laplace_phi = space.laplace_basis(col_in) / sqrt(col_in.shape[0])
del col_in
phi = space.basis(col_bd) / sqrt(col_bd.shape[0])
del col_bd

A_tensor = torch.cat([-laplace_phi,
                      phi], dim=0)
del laplace_phi, phi

A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

um = spsolve(A.T@A, A.T@b)
del A, b, A_tensor, b_tensor
solution = Function(space, 1, torch.from_numpy(um))
end_time = time()

error = solution.estimate_error_tensor(exact_solution, mesh=mesh_err)
print(f"L-2 error: {error.item()}")
print(f"Time: {end_time - start_time}")


# Visualize

from matplotlib import pyplot as plt
fig = plt.figure("RFM for 2d poisson equation")

axes = fig.add_subplot(221)
qm = solution.diff(exact_solution).add_pcolor(axes, box=[0, 1, 0, 1], nums=[80, 80])
axes.set_xlabel('x')
axes.set_ylabel('y')
fig.colorbar(qm)

axes = fig.add_subplot(222, projection='3d')
solution.add_surface(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')

axes = fig.add_subplot(223)
axes.scatter(space.partitions[0].space.frequency[:, 0], um[space.partition_basis_slice(0)])
axes.set_xlabel('FREQ')
axes.set_ylabel('LOAD')
axes.set_title('PART-0')

axes = fig.add_subplot(224)
axes.scatter(space.partitions[1].space.frequency[:, 1], um[space.partition_basis_slice(1)])
axes.set_xlabel('FREQ')
axes.set_ylabel('LOAD')
axes.set_title('PART-1')

plt.show()
