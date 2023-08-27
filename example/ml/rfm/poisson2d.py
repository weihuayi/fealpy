"""
Poisson 2d Example for RFM
"""
from time import time
from math import sqrt

import torch
from torch import Tensor, cos
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.ml.modules import RandomFeaturePoUSpace, PoUSin, Cos, Function, RandomFeatureSpace
from fealpy.mesh import UniformMesh2d, TriangleMesh

NEW_BASIS = False
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
Jn = 64

EXTC = 50
HC = 1/EXTC

start_time = time()

mesh = UniformMesh2d((0, EXT, 0, EXT), (H, H), origin=(0, 0))
node = torch.from_numpy(mesh.entity('node')).clone()
space = RandomFeaturePoUSpace(2, Jn, Cos(), PoUSin(), centers=node, radius=H/2,
                              bound=(PI, PI), print_status=True)

def init_freq(model):
    if isinstance(model, RandomFeatureSpace):
        nf = model.number_of_basis()
        # space.frequency[:] = sqrt(K**2/2)
        model.frequency[:nf//4, 0] = PI/2
        model.frequency[nf//4:nf//4*2, 1] = PI/2
        model.frequency[nf//4*2:nf//4*3, :] = PI/2
        model.frequency[nf//4*3:, 0] = PI/2
        model.frequency[nf//4*3:, 1] = -PI/2

space.apply(init_freq)

if not NEW_BASIS:
    try:
        state_dict = torch.load("rfspace_poisson2d.pth")
        space.load_state_dict(state_dict)
    except:
        pass

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
solution = Function(space, torch.from_numpy(um))
end_time = time()

error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
print(f"L-2 error: {error.data}")
print(f"Time: {end_time - start_time}")

if NEW_BASIS:
    state_dict = space.state_dict()
    torch.save(state_dict, "rfspace_poisson2d.pth")

# Visualize

from matplotlib import pyplot as plt
fig = plt.figure("RFM for 2d poisson equation")

axes = fig.add_subplot(111)
qm = solution.diff(real_solution).add_pcolor(axes, box=[0, 1, 0, 1], nums=[40, 40])
axes.set_xlabel('x')
axes.set_ylabel('y')
fig.colorbar(qm)

plt.show()
