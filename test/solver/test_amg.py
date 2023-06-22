import numpy as np

from fealpy.solver import AMGSolver

import ipdb


def test_amg():
    from fealpy.pde.poisson_2d import CosCosData
    from fealpy.mesh import TriangleMesh 
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.fem import ScalarLaplaceIntegrator 
    from fealpy.fem import ScalarSourceIntegrator
    from fealpy.fem import BilinearForm
    from fealpy.fem import LinearForm
    from fealpy.fem import DirichletBC
    
    pde = CosCosData()
    domain = pde.domain()

    mesh = TriangleMesh.from_box(domain, nx=100, ny=100)

    space = LagrangeFESpace(mesh, p=1)

    bform = BilinearForm(space)
    bform.add_domain_integrator(ScalarLaplaceIntegrator(q=3))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=3))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)

    solver = AMGSolver()
    #ipdb.set_trace()
    solver.setup(A)
    solver.print()



if __name__ == "__main__":
    test_amg()
