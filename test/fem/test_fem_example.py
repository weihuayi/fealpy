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
    p = 3

    em = np.zeros((2, 4), dtype=np.float64)

    for i in range(maxit):
        mesh = IntervalMesh.from_interval_domain(domain, nx=nx * 2**i)
        space = Space(mesh, p=p)

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

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    assert np.abs(ratio[0, -1] - 2**(p+1)) < 0.1
    assert np.abs(ratio[1, -1] - 2**p) < 0.1

def test_triangle_mesh():
    from fealpy.pde.elliptic_2d import SinSinPDEData as PDE
    from fealpy.mesh import TriangleMesh 
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import DiffusionIntegrator
    from fealpy.fem import ScalarSourceIntegrator
    from fealpy.fem import LinearForm
    from fealpy.fem import DirichletBC

    from fealpy.functionspace import LagrangeFiniteElementSpace as OSpace 
    from fealpy.boundarycondition import DirichletBC as DBC


    pde = PDE()
    domain = pde.domain()
    nx = 10 
    ny = 10 
    maxit = 4
    p = 2 # TODO: 检查

    em = np.zeros((2, 4), dtype=np.float64)

    for i in range(maxit):
        mesh = TriangleMesh.from_unit_square(nx=nx*2**i, ny=ny*2**i)
        space = Space(mesh, p=p)

        bc = DirichletBC(space, pde.dirichlet)
        
        bform = BilinearForm(space)
        bform.add_domain_integrator(DiffusionIntegrator())
        bform.assembly()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        lform.assembly()

        A = bform.get_matrix()
        f = lform.get_vector() 

        ospace = OSpace(mesh, p=p)
        obc = DBC(ospace, pde.dirichlet)
        B = ospace.stiff_matrix()
        e = ospace.source_vector(pde.source)

        np.testing.assert_array_almost_equal(A.toarray(), B.toarray())
        np.testing.assert_array_almost_equal(f, e)

        #ipdb.set_trace()
        uh = space.function()
        A, f = bc.apply(A, f, uh)

        B, e = obc.apply(B, e)

        np.testing.assert_array_almost_equal(f, e)
        np.testing.assert_array_almost_equal(A.toarray(), B.toarray())

        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)

    ratio = em[:, 0:-1]/em[:, 1:]
    print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1)) < 0.1
    assert np.abs(ratio[1, -1] - 2**p) < 0.1

def test_tetrahedron_mesh():
    from fealpy.pde.elliptic_3d import SinSinSinPDEData as PDE
    from fealpy.mesh import TetrahedronMesh 
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import DiffusionIntegrator
    from fealpy.fem import ScalarSourceIntegrator
    from fealpy.fem import LinearForm
    from fealpy.fem import DirichletBC

    from fealpy.functionspace import LagrangeFiniteElementSpace as OSpace 
    from fealpy.boundarycondition import DirichletBC as DBC


    pde = PDE()
    domain = pde.domain()
    nx = 5 
    ny = 5 
    nz = 5
    maxit = 4
    p = 1

    em =  np.zeros((2, 4), dtype=np.float64)

    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
    mesh.uniform_refine(n=3)
    for i in range(maxit):
        space = Space(mesh, p=p)
        bc = DirichletBC(space, pde.dirichlet)
        
        bform = BilinearForm(space)
        bform.add_domain_integrator(DiffusionIntegrator())
        bform.assembly()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(pde.source))
        lform.assembly()

        ipdb.set_trace()
        A = bform.get_matrix()
        f = lform.get_vector() 

        ospace = OSpace(mesh, p=p)
        obc = DBC(ospace, pde.dirichlet)
        B = ospace.stiff_matrix()
        e = ospace.source_vector(pde.source)

        np.testing.assert_array_almost_equal(A.toarray(), B.toarray())
        np.testing.assert_array_almost_equal(f, e)

        uh = space.function()
        A, f = bc.apply(A, f, uh)

        B, e = obc.apply(B, e)

        np.testing.assert_array_almost_equal(f, e)
        np.testing.assert_array_almost_equal(A.toarray(), B.toarray())

        uh[:] = spsolve(A, f)

        em[0, i] = mesh.error(pde.solution, uh, q=p+3)
        em[1, i] = mesh.error(pde.gradient, uh.grad_value, q=p+3)
        if i < maxit-1:
            mesh.uniform_refine()

    ratio = em[:, 0:-1]/em[:, 1:]
    print(ratio)
    assert np.abs(ratio[0, -1] - 2**(p+1)) < 0.1
    assert np.abs(ratio[1, -1] - 2**p) < 0.1

if __name__ == "__main__":
    #test_triangle_mesh()
    test_tetrahedron_mesh()

