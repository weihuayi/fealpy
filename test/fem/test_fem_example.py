import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest
import ipdb

from fealpy.fem import BilinearForm
from fealpy.quadrature import FEMeshIntegralAlg

def test_interval_mesh():
    from fealpy.pde.elliptic_1d import SinPDEData as PDE
    from fealpy.mesh import IntervalMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import DiffusionIntegrator
    from fealpy.fem import ScalarSourceIntegrator
    from fealpy.fem import LinearForm
    from fealpy.fem import DirichletBC


    pde = PDE()
    domain = pde.domain()
    nx = 10
    maxit = 4

    em = np.zeros((2, 4), dtype=np.float64)

    for i in range(maxit):
        mesh = IntervalMesh.from_interval_domain(domain, nx=nx * 2**i)
        space = Space(mesh, p=1)

        bc = DirichletBC(space, pde.dirichlet)
        
        bform = BilinearForm(space)
        bform.add_domain_integrator(DiffusionIntegrator())
        bform.assembly()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        lform.assembly()

        A = bform.get_matrix()
        f = lform.get_vector() 

        uh = space.function()
        A, f = bc.apply(A, f, uh)
        uh[:] = spsolve(A, f)

        ipdb.set_trace()
        em[0, i] = mesh.error(pde.solution, uh, q=3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=3)

    print(em)


if __name__ == "__main__":
    test_interval_mesh()

