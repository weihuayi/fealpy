import time

import torch
from torch import Tensor, sin
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt

from fealpy.ml.modules import RandomFeatureSpace, Cos, PoUSpace, PoUSin
from fealpy.ml.operators import Form, ScalerDiffusion, ScalerMass
from fealpy.ml.sampler import ISampler, BoxBoundarySampler
from fealpy.mesh import UniformMesh2d, TriangleMesh


"""
    \Delta u(x,y) + k**2 * u(x,y) = 0 ,                            (x,y)\in \Omega
    u(x,y) = \sin(\sqrt{k**2/2} * x + \sqrt{k**2/2} * y) ,         (x,y)\in \partial\Omega
"""

# Number of waves
K = 10
k = torch.tensor(K, dtype=torch.float64)

# Exact solution
def real_solution(p: Tensor):
    x = p[:, 0:1]
    y = p[:, 1:2]
    return sin(torch.sqrt(k**2/2) * x + torch.sqrt(k**2/2) * y)

# Boundary conditions
def boundary(p: Tensor):
    return real_solution(p)


start_time = time.time()

def factory(_: int):
    return RandomFeatureSpace(2, 96, Cos(), bound=(2.5, torch.pi))

mesh = UniformMesh2d([0, 4, 0, 4], [2/4, 2/4], origin=(-1, -1))
space = PoUSpace.from_uniform_mesh(
    factory, mesh, 'node', pou=PoUSin(), print_status=True
)

col_in = ISampler([-1, 1, -1, 1], 'random').run(22801)
col_bd = BoxBoundarySampler([-1, -1], [1, 1], 'linspace').run(181, 181)

form = Form(space)
form.add(col_in, (ScalerDiffusion(-1.), ScalerMass(k**2)))
form.add(col_bd, ScalerMass(), boundary)
A, b = form.assembly(rescale=100.)

um = spsolve(A, b)
solution = space.function(torch.from_numpy(um))
end_time = time.time()

# L2 error

mesh_err = TriangleMesh.from_box([-1, 1, -1, 1], nx=50, ny=50)
error = solution.estimate_error_tensor(real_solution, mesh=mesh_err)
training_time = end_time - start_time
print(f"L-2 error: {error.item()}")
print("Time: ", training_time, "s")

# Visualization

fig = plt.figure()

axes = fig.add_subplot(121)
solution.add_pcolor(axes, box=[-1, 1, -1, 1], nums=[200, 200])
axes = fig.add_subplot(122)
qm = solution.diff(real_solution).add_pcolor(axes, box=[-1, 1, -1, 1], nums=[200, 200])
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('diff')
fig.colorbar(qm)

plt.show()
