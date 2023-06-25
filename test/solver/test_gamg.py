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
    
    pde = CosCosData()
    domain = pde.domain()

    mesh = TriangleMesh.from_box(domain, nx=400, ny=400)

    space = LagrangeFESpace(mesh, p=3)

    bform = BilinearForm(space)
    bform.add_domain_integrator(ScalarLaplaceIntegrator(q=3))
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(pde.source, q=4))
    F = lform.assembly()

    bc = DirichletBC(space, pde.dirichlet) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)

    L = sp.tril(A, format='csr')
    U = sp.triu(A, format='csr')
    D = A.diagonal()
    #ipdb.set_trace()
    P = mesh.prolongation_matrix(1, 3)
    R = P.T.tocsr()
    A1 = R@A@P

    solver = GAMGSolver(ptype='W', sstep=3)
    solver.amg_setup(A1)

    solver.A.insert(0, A)
    solver.L.insert(0, L)
    solver.U.insert(0, U)
    solver.D.insert(0, D)
    solver.P.insert(0, P)
    solver.R.insert(0, R)
    solver.print()
    uh[:] = solver.solve(F)



if __name__ == "__main__":
    test_gamg()
