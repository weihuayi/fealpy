import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest
import ipdb

from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import ScalarNeumannBCIntegrator

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 8, 4), (2, 6, 4), (3, 4, 4), (4, 4, 4)])
def test_triangle_mesh(p, n, maxit):
    from fealpy.pde.poisson_2d import CosCosData as PDE
    from fealpy.mesh import TriangleMesh 

    pde = PDE()
    domain = pde.domain()
    em = np.zeros((2, maxit), dtype=np.float64)

    for i in range(maxit):
        mesh = TriangleMesh.from_unit_square(nx=n*2**i, ny=n*2**i)
        space = Space(mesh, p=p)
        
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        bform.assembly()
        A = bform.get_matrix()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        bi = ScalarNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary)
        lform.add_boundary_integrator(bi)
        lform.assembly()
        f = lform.get_vector() 
        
        uh = space.function()
        bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        A, f = bc.apply(A, f, uh)

        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    print(em)
    print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1))/2**(p+1) <0.2
    assert np.abs(ratio[1, -1] - 2**p)/2**p <0.2

if __name__=="__main__":
    test_triangle_mesh(p=4, n=4, maxit=4)
