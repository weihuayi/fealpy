#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from fealpy.pde.elliptic_1d import SinPDEData 
from fealpy.mesh import UniformMesh1d

pde = SinPDEData()
domain = pde.domain()

nx = 10
maxit = 4
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

em = np.zeros((3, maxit), dtype=np.float64)
egradm = np.zeros((3, maxit), dtype=np.float64)

for i in range(maxit):
    uh = mesh.function()
    A = mesh.laplace_operator()
    f = mesh.interpolate(pde.source)
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f, uh=uh)
    uh[:] = spsolve(A, f)
    grad_uh = mesh.gradient(f=uh[:], order=1)
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    egradm[0, i], egradm[1, i], egradm[2, i] = mesh.error(pde.gradient, grad_uh)

    if i < maxit - 1:
        mesh.uniform_refine()

print(em[:, 0:-1]/em[:, 1:])
print(egradm[:, 0:-1]/egradm[:, 1:])
