import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest

from fealpy.fem import BilinearForm

def test_interval_mesh():
    from fealpy.pde.elliptic_1d import SinPDEData as PDE
    from fealpy.mesh import IntervalMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import DiffusionIntegrator
    from fealpy.fem import LinearForm
    from fealpy.fem import DirichletBC


    pde = PDE()
    domain = pde.domain()
    mesh = IntervalMesh.from_interval_domain(domain, nx=10)
    space = LagrangeFESpace(mesh, p=1)
    
    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator())
    bform.assembly()

    K = bform.get_matrix()

def test_triangle_mesh():

    from fealpy.mesh import TriangleMesh 
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace 
    from fealpy.fem import DiffusionIntegrator

    p=3
    mesh = TriangleMesh.from_one_triangle()
    mesh.uniform_refine()
    space = Space(mesh, p=p)

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator())
    bform.assembly()

    ospace = OldSpace(mesh, p=p)

    A = bform.get_matrix()

    B = ospace.stiff_matrix()

    np.testing.assert_array_almost_equal(A.toarray(), B.toarray())

def test_tetrahedron_mesh():

    from fealpy.mesh import TetrahedronMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace 
    from fealpy.fem import DiffusionIntegrator

    p=1
    mesh = TetrahedronMesh.from_one_tetrahedron()
    mesh.uniform_refine(n=1)
    space = Space(mesh, p=p)

    bform = BilinearForm(space)
    bform.add_domain_integrator(DiffusionIntegrator())
    bform.assembly()

    ospace = OldSpace(mesh, p=p)

    A = bform.get_matrix()

    B = ospace.stiff_matrix()

    np.testing.assert_array_almost_equal(A.toarray(), B.toarray())

def test_truss_structure():

    from fealpy.mesh import EdgeMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import TrussStructureIntegrator
    from fealpy.fem import DirichletBC
    
    mesh = EdgeMesh.from_tower()
    GD = mesh.geo_dimension()
    space = Space(mesh, p=1, doforder='vdims')

    bform = BilinearForm(GD*(space,))

    E = 1500 # 杨氏模量
    A = 2000 # 横截面积
    bform.add_domain_integrator(TrussStructureIntegrator(E, A))
    bform.assembly()

    K = bform.get_matrix()

    uh = space.function(dim=GD)
    
    # 加载力的条件 
    F = np.zeros((uh.shape[0], GD), dtype=np.float64)
    idx, f = mesh.meshdata['force_bc']
    F[idx] = f 

    idx, disp = mesh.meshdata['disp_bc']
    bc = DirichletBC(space, disp, threshold=idx)
    A, F = bc.apply(K, F.flat, uh)

    uh.flat[:] = spsolve(A, F)
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection='3d') 
    mesh.add_plot(axes)

    mesh.node += 100*uh
    mesh.add_plot(axes, nodecolor='b', cellcolor='m')
    plt.show()

def test_linear_elasticity_model():
    from fealpy.pde.linear_elasticity_model import  BoxDomainData3d as PDE
    from fealpy.mesh import TriangleMesh 
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import LinearElasticityOperatorIntegrator
    from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace 

    pde = PDE()
    domain = pde.domain()
    nx = 10 
    ny = 10 
    p = 1 

#    mesh = TriangleMesh.from_unit_square(nx=nx*2, ny=ny*2)
    mesh = TriangleMesh.from_one_triangle()
    space = Space(mesh, p=p, doforder='sdofs')
    ospace = OldSpace(mesh, p=p)

    GD = mesh.geo_dimension()
    bform = BilinearForm(GD*(space,))
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(lam=pde.lam,
        mu=pde.mu, q=p))

    bform.assembly()

    A = bform.get_matrix()
    B = ospace.linear_elasticity_matrix(pde.lam, pde.mu, q=p)
    print(A.toarray())
    print(B.toarray())

    np.testing.assert_array_almost_equal(A.toarray(), B.toarray())


if __name__ == '__main__':
    test_linear_elasticity_model()
    #test_tetrahedron_mesh()
    #test_triangle_mesh()
    #test_truss_structure()



