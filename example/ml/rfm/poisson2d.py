"""
Poisson 2d Example for RFM
"""

from math import sqrt

import torch
from torch import Tensor, cos
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import RandomFeaturePoUSpace, PoUSin, Cos, RFFunction
from fealpy.mesh import UniformMesh2d, TriangleMesh


PI = torch.pi

def real_solution(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return cos(PI * x) * cos(PI * y)

def boundary(p: Tensor):
    return real_solution(p)

def source(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return 2 * PI**2 * cos(PI * x) * cos(PI * y)


EXT = 1
H = 1/EXT
Jn = 128

EXTC = 50
HC = 1/EXTC


mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
node = torch.from_numpy(mesh.entity('node')).clone()
space = RandomFeaturePoUSpace(2, Jn, Cos(), PoUSin(), centers=node, radius=H/2,
                              bound=(PI, PI), print_status=True)

mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))

mesh_err = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)


laplace_phi = space.L(col_in) / sqrt(col_in.shape[0])
phi = space.U(col_bd) / sqrt(col_bd.shape[0])

A_tensor = torch.cat([-laplace_phi,
                      phi], dim=0)
b_tensor = torch.cat([source(col_in) / sqrt(col_in.shape[0]),
                      boundary(col_bd) / sqrt(col_bd.shape[0])], dim=0)

A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

um = spsolve(A.T@A, A.T@b)
solution = RFFunction(space, torch.from_numpy(um))


error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
print(f"L-2 error: {error.data}")


# Visualize

from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(121, projection='3d')
solution.add_surface(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('phi')

axes = fig.add_subplot(122, projection='3d')
solution.diff(real_solution).add_surface(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('phi')

plt.show()
