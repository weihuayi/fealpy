import numpy as np

import pytest

def test_linear_elasticity_lfem(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData2d
    from fealpy.mesh import TriangleMesh
    from fealpy.functionspace import LagrangeFESpace as Space
    from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace
    from fealpy.fem import LinearElasticityOperatorIntegrator
    from fealpy.fem import BilinearForm
    from fealpy.fem import LinearForm

    pde = BoxDomainData2d()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    space = Space(mesh, p=p)

    ospace = Space(mesh, p=p)

    bform = BilinearForm(space)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu)
    bform.assembly()
    A = bform


