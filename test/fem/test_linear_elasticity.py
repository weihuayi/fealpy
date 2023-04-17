import numpy as np

import pytest

def test_linear_elasticity_lfem(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData2d
    from fealpy.mesh import TriangleMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.fem import LinearElasticityOperatorIntegrator
    from fealpy.fem import VectorSourceIntegrator
    from fealpy.fem import BilinearForm
    from fealpy.fem import LinearForm

    from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace

    pde = BoxDomainData2d()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    
    # 新接口程序
    space = Space(mesh, p=p)
    bform = BilinearForm(space)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
    bform.assembly()
    A = bform.get_matrix()

    # 老接口程序 
    ospace = OldSpace(mesh, p=p)
    uh = space.function(dim=GD) # (NDof, GD)
    A = space.linear_elasticity_matrix(pde.lam, pde.mu, q=p+2)
    F = space.source_vector(pde.source, dim=GD)

    if hasattr(pde, 'neumann'):
        print('neumann')
        bc = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)
        F = bc.apply(F)

    if hasattr(pde, 'dirichlet'):
        print('dirichlet')
        bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
        A, F = bc.apply(A, F, uh)

