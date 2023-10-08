"""
RFM for 2-d Poisson problem

Equation:
  - \\Delta u = f

Exact solution:
  \\exp^{-0.5(x^2 + y^2)}
"""

### STEP 1: Import third-party libraries

from time import time
from math import sqrt

import torch
from torch import Tensor, exp
from torch.nn import init
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import PoUSpace, PoUSin, Cos, Function, RandomFeatureSpace
from fealpy.mesh import UniformMesh2d, TriangleMesh

### STEP 2: Define the PDE, source and boundary condition

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

### STEP 3: Construct function space

EXT = 1
H = 1/EXT
Jn = 162
PI = torch.pi

start_time = time()

def factory(i: int):
    sp = RandomFeatureSpace(2, Jn, Cos(), bound=(1, PI))
    init.normal_(sp.frequency, 0.0, PI/2)
    return sp

mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
node = torch.from_numpy(mesh.entity('node')).clone()
space = PoUSpace(factory, pou=PoUSin(), centers=node, radius=H/2, print_status=True)
del mesh

### STEP 4: Get collocation points

EXTC = 100
HC = 1/EXTC

mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))
del _bd_node, mesh_col

### STEP 5: Assemble matrix

b_tensor = torch.cat([source(col_in) / sqrt(col_in.shape[0]),
                      boundary(col_bd) / sqrt(col_bd.shape[0])], dim=0)

laplace_phi = space.laplace_basis(col_in) / sqrt(col_in.shape[0])
phi = space.basis(col_bd) / sqrt(col_bd.shape[0])
del col_in, col_bd

A_tensor = torch.cat([-laplace_phi,
                      phi], dim=0)
del laplace_phi, phi

A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

### STEP 6: Solve the problem, and construct function in space

um = spsolve(A.T@A, A.T@b)
del A, b, A_tensor, b_tensor
solution = Function(space, torch.from_numpy(um))
end_time = time()

### STEP 7: Estimate error and visualize

mesh_err = TriangleMesh.from_box([0, 1, 0, 1], nx=50, ny=50)
error = solution.estimate_error_tensor(exact_solution, mesh=mesh_err)
print(f"L-2 error: {error.item()}")
print(f"Time: {end_time - start_time}")


from matplotlib import pyplot as plt
fig = plt.figure("RFM for 2d poisson equation")

axes = fig.add_subplot(121)
qm = solution.diff(exact_solution).add_pcolor(axes, box=[0, 1, 0, 1], nums=[80, 80])
axes.set_xlabel('x')
axes.set_ylabel('y')
fig.colorbar(qm)

axes = fig.add_subplot(122, projection='3d')
solution.add_surface(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')

plt.show()
