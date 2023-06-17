import numpy as np

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from scipy.sparse.linalg import spsolve

maxit = 6
p = 2
pde = CosCosData()
domain = pde.domain()
h = 0.2
error = np.zeros(maxit)
for i in range(maxit):
    print(i)
    mesh = triangle(domain, h, meshtype='polygon')
    space = ConformingVirtualElementSpace2d(mesh, p=p, q=p+3)
    M = space.mass_matrix()
    F = space.source_vector(pde.solution)

    uh = space.function()
    uh[:] = spsolve(M, F).reshape(-1)

    sh = space.project_to_smspace(uh)

    def efun(x, cellidx=None):
        return (pde.solution(x) - sh(x, cellidx))**2

    error[i] = space.integralalg.error(efun, power=np.sqrt)
    h /= 2

print(error[:-1]/error[1:])

