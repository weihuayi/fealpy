"""
Heat 1d Example for RFM
"""

from math import sqrt

import torch
from torch import Tensor, exp, sin
from torch.nn import init
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import PoUSpace, PoUSin, Cos, Function, RandomFeatureSpace
from fealpy.mesh import UniformMesh2d, TriangleMesh


PI = torch.pi

def real_solution(p: Tensor):
    t = p[:, 0:1]
    x = p[:, 1:2]
    return sin(2*PI*x) * exp(-0.2 * PI**2 * t)

def boundary(p: Tensor):
    return real_solution(p)

def source(p: Tensor):
    return torch.zeros((p.shape[0], 1), dtype=p.dtype, device=p.device)


EXT = 1
H = 1/EXT
Jn = 32

EXTC = 50
HC = 1/EXTC

def factory(i: int):
    return RandomFeatureSpace(2, Jn, Cos(), bound=(PI, PI))

mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
node = torch.from_numpy(mesh.entity('node')).clone()
space = PoUSpace(factory, pou=PoUSin(), centers=node, radius=H/2, print_status=True)

def init_freq(model):
    if isinstance(model, RandomFeatureSpace):
        with torch.no_grad():
            model.frequency[:, 1] = PI
            init.normal_(model.frequency[:, 0], 0, 0.2*PI**2)

space.apply(init_freq)

mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(0, 0))
_bd_node = mesh_col.ds.boundary_node_flag()
col_in = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
col_bd = torch.from_numpy(mesh_col.entity('node', index=_bd_node))[:-EXTC-1, :]

mesh_err = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)


QI = sqrt(col_in.shape[0])
QB = sqrt(col_bd.shape[0])

phi_t = space.derivative_basis(col_in, 0) / QI
phi_xx = space.derivative_basis(col_in, 1, 1) / QI
phi = space.basis(col_bd) / QB

A_tensor = torch.cat([phi_t - 0.05*phi_xx,
                      phi], dim=0)
b_tensor = torch.cat([source(col_in) / QI,
                      boundary(col_bd) / QB], dim=0)

A = csr_matrix(A_tensor.cpu().numpy())
b = csr_matrix(b_tensor.cpu().numpy())

um = spsolve(A.T@A, A.T@b)
solution = Function(space, torch.from_numpy(um))


error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
print(f"L-2 error: {error.data}")


# Visualize

from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(121, projection='3d')
solution.add_surface(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('t')
axes.set_ylabel('x')
axes.set_zlabel('phi')

axes = fig.add_subplot(122)
qm = solution.diff(real_solution).add_pcolor(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('t')
axes.set_ylabel('x')
fig.colorbar(qm)

plt.show()
