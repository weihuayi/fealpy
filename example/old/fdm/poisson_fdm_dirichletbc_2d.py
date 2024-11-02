import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.elliptic_2d import SinSinPDEData
from fealpy.mesh import UniformMesh2d

pde = SinSinPDEData()
domain = pde.domain()

maxit = 4
nx = 10
ny = 10
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))
em = np.zeros((3, maxit), dtype=np.float64)
for i in range(maxit):
    uh = mesh.function()
    A = mesh.laplace_operator()
    f = mesh.interpolate(pde.source)
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f, uh=uh)
    uh.flat[:] = spsolve(A, f)
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    if i < maxit - 1:
        mesh.uniform_refine()
print(em[:, 0:-1]/em[:, 1:])
