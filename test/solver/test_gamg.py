import numpy as np
import scipy.sparse as sp

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
   
    p = 6
    pde = CosCosData()
    domain = pde.domain()

    mesh = TriangleMesh.from_box(domain, nx=100, ny=100)

    space = LagrangeFESpace(mesh, p=p)

    bform = BilinearForm(space)
    bform.add_domain_integrator(ScalarLaplaceIntegrator(q=p+2))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=p+2))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)

    solver = GAMGSolver(ptype='W', sstep=3)
    solver.setup(A, space=space, cdegree=[1, 3])
    solver.print()
    uh[:] = solver.solve(F)



if __name__ == "__main__":
    test_gamg()
