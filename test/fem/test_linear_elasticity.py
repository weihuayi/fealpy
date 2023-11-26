import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import ipdb
import pytest

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import VectorNeumannBCIntegrator

from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace
from fealpy.boundarycondition import DirichletBC as OldDirichletBC
from fealpy.boundarycondition import NeumannBC as OldNeumannBC

def test_linear_elasticity_lfem_2d(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData2d

    pde = BoxDomainData2d()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    GD = mesh.geo_dimension()
    print("GD:", GD)
    NN = mesh.number_of_nodes()
    print("NN:", NN)

    ospace = OldSpace(mesh, p=p)
    ouh = ospace.function(dim=GD)
    
    # 新接口程序
    # 构建双线性型，表示问题的微分形式
    space = Space(mesh, p=p, doforder='vdims')
    uh = space.function(dim=GD)
    vspace = GD*(space, ) # 把标量空间张成向量空间
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
    bform.assembly()

    # 构建单线性型，表示问题的源项
    lform = LinearForm(vspace)
    lform.add_domain_integrator(VectorSourceIntegrator(pde.source, q=1))
    if hasattr(pde, 'neumann'):
        bi = VectorNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=1)
        lform.add_boundary_integrator(bi)
    lform.assembly()

    A = bform.get_matrix()
    F = lform.get_vector()
    idx = np.r_['0', np.arange(0, NN*2, 2), np.arange(1, NN*2, 2)]
    B = A[idx, :][:, idx]

    

    oA = ospace.linear_elasticity_matrix(pde.lam, pde.mu, q=p+2)
    oF = ospace.source_vector(pde.source, dim=GD)

    if hasattr(pde, 'neumann'):
        print('neumann')
        bc = OldNeumannBC(ospace, pde.neumann, threshold=pde.is_neumann_boundary)
        oF = bc.apply(oF)

    if hasattr(pde, 'dirichlet'):
        bc = DirichletBC(vspace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        print("bc:", bc)
        print("A:", A.shape)
        print("F:", F.shape)
        print("uh:", uh.shape)
        A, F = bc.apply(A, F, uh)

    if hasattr(pde, 'dirichlet'):
        print('dirichlet')
        bc = OldDirichletBC(ospace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        oA, oF = bc.apply(oA, oF, ouh)

    #np.testing.assert_array_almost_equal(F, oF)

    ouh.T.flat = spsolve(oA, oF)
    uh.flat = spsolve(A, F)

    # 画出原始网格
    mesh.add_plot(plt)

    # 画出变形网格
    mesh.node += 100*uh[:NN]
    mesh.add_plot(plt)

    plt.show()


def test_linear_elasticity_lfem_3d(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData3d

    pde = BoxDomainData3d()
    domain = pde.domain()
    mesh = pde.init_mesh(n=n)
    GD = mesh.geo_dimension()
    NN = mesh.number_of_nodes()
    print("NN:", NN)

    ospace = OldSpace(mesh, p=p)
    ouh = ospace.function(dim=GD)
    
    # 新接口程序
    space = Space(mesh, p=p, doforder='sdofs')
    uh = space.function(dim=GD)
    vspace = GD*(space, ) # 把标量空间张成向量空间
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
    bform.assembly()

    lform = LinearForm(vspace)
    lform.add_domain_integrator(VectorSourceIntegrator(pde.source, q=1))
    if hasattr(pde, 'neumann'):
        bi = VectorNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=1)
        lform.add_boundary_integrator(bi)
    lform.assembly()

    A = bform.get_matrix()
    F = lform.get_vector()
    oA = ospace.linear_elasticity_matrix(pde.lam, pde.mu, q=p+2)
    oF = ospace.source_vector(pde.source, dim=GD)

    if hasattr(pde, 'neumann'):
        print('neumann')
        bc = OldNeumannBC(ospace, pde.neumann, threshold=pde.is_neumann_boundary)
        oF = bc.apply(oF)
    
    np.testing.assert_array_almost_equal(A.toarray(), oA.toarray())
    np.testing.assert_array_almost_equal(F, oF.T.flat)


    if hasattr(pde, 'dirichlet'):
        bc = DirichletBC(vspace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        A, F = bc.apply(A, F, uh)

    if hasattr(pde, 'dirichlet'):
        print('dirichlet')
        bc = OldDirichletBC(ospace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        oA, oF = bc.apply(oA, oF, ouh)
    np.testing.assert_array_almost_equal(A.toarray(), oA.toarray())
    np.testing.assert_array_almost_equal(F, oF)

    ouh.T.flat = spsolve(oA, oF)
    uh.flat = spsolve(A, F)

    # 画出原始网格
    mesh.add_plot(plt)

    # 画出变形网格
    mesh.node += 100*uh[:, :NN].T
    mesh.add_plot(plt)

    plt.show()


if __name__ == "__main__":
    test_linear_elasticity_lfem_2d(1, 10)
    #test_linear_elasticity_lfem_3d(1, 1)
