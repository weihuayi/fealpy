import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest
import ipdb


from fealpy.functionspace import LagrangeFESpace as Space

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import ScalarNeumannBCIntegrator

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 10, 4), (2, 8, 4), (3, 6, 4), (4, 4, 4)])
def test_dirichlet_bc_on_interval_mesh(p, n, maxit):
    from fealpy.pde.elliptic_1d import SinPDEData as PDE
    from fealpy.mesh import IntervalMesh

    pde = PDE()
    domain = pde.domain()
    em = np.zeros((2, maxit), dtype=np.float64)

    for i in range(maxit):
        mesh = IntervalMesh.from_interval_domain(domain, nx=n * 2**i)
        space = Space(mesh, p=p)

        
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        bform.assembly()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        lform.assembly()

        A = bform.get_matrix()
        f = lform.get_vector() 

        bc = DirichletBC(space, pde.dirichlet)
        A, f = bc.apply(A, f, uh)
        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    print(em)
    print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1)) < 0.3
    assert np.abs(ratio[1, -1] - 2**p) < 0.3

def test_dirichlet_and_neumann_bc_on_interval_mesh(p, n, maxit):
    from fealpy.pde.elliptic_1d import SinPDEData as PDE
    from fealpy.mesh import IntervalMesh

    pde = PDE()
    domain = pde.domain()
    em = np.zeros((2, maxit), dtype=np.float64)

    for i in range(maxit):
        mesh = IntervalMesh.from_interval_domain(domain, nx=n * 2**i)
        space = Space(mesh, p=p)

        
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        bform.assembly()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        bi = ScalarNeumannBCIntegrator(pde.neumann,
                threshold=pde.is_neumann_boundary)
        lform.add_boundary_integrator(bi)
        lform.assembly()

        A = bform.get_matrix()
        f = lform.get_vector() 

        bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        A, f = bc.apply(A, f, uh)
        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    print(em)
    print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1)) < 0.3
    assert np.abs(ratio[1, -1] - 2**p) < 0.3

if __name__ == "__main__":
    #test_interval_mesh(1, 10, 4)
    #test_interval_mesh(2, 8, 4)
    #test_interval_mesh(3, 6, 4)
    test_dirichlet_bc_on_interval_mesh(4, 4, 4)


