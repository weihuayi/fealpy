import numpy as np
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

def test_linear_elasticity_lfem(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData2d

    from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace
    from fealpy.boundarycondition import NeumannBC as OldNeumannBC
    from fealpy.boundarycondition import DirichletBC as OldDirichletBC

    pde = BoxDomainData2d()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    mesh = TriangleMesh.from_one_triangle()
    GD = mesh.geo_dimension()
    
    # 新接口程序
    space = Space(mesh, p=p)
    uh = space.function(dim=GD)
    vspace = GD*(space, ) # 把标量空间张成向量空间
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
    bform.assembly()

    lform = LinearForm(vspace)
    lform.add_domain_integrator(VectorSourceIntegrator(pde.source, q=1))
    bint = VectorNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=1)
    lform.add_boundary_integrator(bint)
    lform.assembly()

    A = bform.get_matrix()
    F = lform.get_vector()

    bc = DirichletBC(vspace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    A, F = bc.apply(A, F, uh)

    # 老接口程序 
    ospace = OldSpace(mesh, p=p)
    ouh = ospace.function(dim=GD) # (NDof, GD)
    oA = ospace.linear_elasticity_matrix(pde.lam, pde.mu, q=p+2)
    oF = ospace.source_vector(pde.source, dim=GD).T.flat


    if hasattr(pde, 'neumann'):
        print('neumann')
        bc = OldNeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)
        OF = bc.apply(OF)

    if hasattr(pde, 'dirichlet'):
        print('dirichlet')
        bc = OldDirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        oA, oF = bc.apply(oA, oF, ouh)

    np.testing.assert_array_almost_equal(A.toarray(), oA.toarray())
    np.testing.assert_array_almost_equal(F, oF)


if __name__ == "__main__":
    test_linear_elasticity_lfem(1, 10)
