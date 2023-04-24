import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest
import ipdb


from fealpy.mesh import TetrahedronMesh 
from fealpy.pde.elliptic_3d import SinSinSinPDEData as PDE

from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import ScalarNeumannBCIntegrator as NeumannBC

from fealpy.boundarycondition import NeumannBC as OldNeumannBC
from fealpy.boundarycondition import DirichletBC as OldDirichletBC

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 2, 4), (2, 2, 4), (3, 2, 4), (4, 2, 4)])
def test_lfe_dirichlte_bc_on_tetrahedron_mesh(p, n, maxit):
    pde = PDE()
    domain = pde.domain()
    em =  np.zeros((2, maxit), dtype=np.float64)

    for i in range(maxit):
        mesh = TetrahedronMesh.from_unit_cube(nx=n * 2**i, ny=n * 2**i, nz= n * 2**i)
        space = Space(mesh, p=p)
        
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        bform.assembly()
        A = bform.get_matrix()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        lform.assembly()
        f = lform.get_vector() 

        ospace = OldSpace(mesh, p=p)
        OA = ospace.stiff_matrix()
        Of = ospace.source_vector(pde.source)

        bc = DirichletBC(space, pde.dirichlet)
        uh = space.function()
        A, f = bc.apply(A, f, uh)

        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    #print(em)
    #print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1))/2**(p+1) < 0.02
    assert np.abs(ratio[1, -1] - 2**p)/2**p < 0.02

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 2, 4), (2, 2, 4), (3, 2, 4), (4, 2, 4)])
def test_lfe_dirichlte_and_neumann_bc_on_tetrahedron_mesh(p, n, maxit):
    pde = PDE()
    domain = pde.domain()
    em =  np.zeros((2, maxit), dtype=np.float64)

    for i in range(maxit):
        mesh = TetrahedronMesh.from_unit_cube(nx=n * 2**i, ny=n * 2**i, nz= n * 2**i)
        bdtype  = mesh.meshdata
        neuBC = np.r_[bdtype['leftface'], bdtype['rightface'], bdtype['frontface'], bdtype['backface']]
        dirBC = np.r_[bdtype['upface'], bdtype['bottomface']]

        space = Space(mesh, p=p)

        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        bform.assembly()
        A = bform.get_matrix()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        lform.assembly()
        f = lform.get_vector() 

        bc2 = NeumannBC(pde.neumann, threshold=neuBC)
        bc2.assembly_face_vector(space, out=f)
        
        uh = space.function()
        bc1 = DirichletBC(space, pde.dirichlet, threshold=dirBC)
        A, f = bc1.apply(A, f, uh)

        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    #print(em)
    #print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1))/2**(p+1) < 0.02
    assert np.abs(ratio[1, -1] - 2**p)/2**p < 0.02
if __name__ == "__main__":
    #test_lfe_dirichlte_bc_on_tetrahedron_mesh(2, 1, 5)
    test_lfe_dirichlte_and_neumann_bc_on_tetrahedron_mesh(2, 1, 5)

