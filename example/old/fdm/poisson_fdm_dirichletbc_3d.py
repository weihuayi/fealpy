import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from fealpy.pde.elliptic_3d import SinSinSinPDEData 
from fealpy.mesh import UniformMesh3d

pde = SinSinSinPDEData()
domain = pde.domain()

nx = 10
ny = 10
nz = 10
maxit = 2
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
hz = (domain[5] - domain[4])/nz
mesh = UniformMesh3d(
        [0, nx, 0, ny, 0, nz], 
        h=(hx, hy, hz), 
        origin=(domain[0], domain[2], domain[4]))
em = np.zeros((3, maxit), dtype=np.float64)
for i in range(maxit):
    uh = mesh.function()
    A = mesh.laplace_operator()
    f = mesh.interpolate(pde.source)
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f, uh=uh)
    uh.flat[:] = spsolve(A, f) # TODO: 使用 spsolve 无法求解太大规模的洗漱矩阵
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    if i < maxit - 1:
        mesh.uniform_refine()
print(em[:, 0:-1]/em[:, 1:])
