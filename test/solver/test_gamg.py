import numpy as np

from fealpy.solver import GAMGSolver

import ipdb


def test_gamg():
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

    mesh = TriangleMesh.from_box(domain, nx=400, ny=400)

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

    solver = GAMGSolver(ptype='W', sstep=3)
    solver.amg_setup(A)
    solver.print()
    #ipdb.set_trace()
    uh[:] = solver.solve(F)



if __name__ == "__main__":
    test_gamg()
