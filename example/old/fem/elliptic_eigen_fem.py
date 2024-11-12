
import numpy as np
from scipy.sparse.linalg import eigs, LinearOperator

from fealpy.mesh import TriangleMesh

from fealpy.functionspace import LagrangeFESpace

from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import BilinearForm

from fealpy.solver import GAMGSolver
from fealpy.solver import MatlabSolver

import transplant

matalb = transplant.Matlab()
msolver = MatlabSolver(matlab)

def linear_operator(x):
    y = M@x
    z = solver.solve(y)
    return z

h = 0.2
maxit = 1
k=5
es = np.zeros((maxit, k), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.float64)
vertices = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]], dtype=np.float64)

for i in range(maxit):
    mesh = TriangleMesh.from_polygon_gmsh(vertices, h/(2**i))

    space = LagrangeFESpace(mesh, p=1)

    isFreeDof = ~space.is_boundary_dof()
    bf0 = BilinearForm(space)
    bf0.add_domain_integrator(ScalarDiffusionIntegrator(q=3))
    A = bf0.assembly()

    bf1 = BilinearForm(space)
    bf1.add_domain_integrator(ScalarMassIntegrator(q=3))
    M = bf1.assembly() 

    A = A[isFreeDof, :][:, isFreeDof].tocsr()
    M = M[isFreeDof, :][:, isFreeDof].tocsr()
    NN = A.shape[0]

    if False:
        solver = GAMGSolver(ptype='W', sstep=2)
        solver.setup(A)

        P = LinearOperator((NN, NN), matvec=linear_operator)
        vals, vecs = eigs(P, k=5)
    else:
        vecs, vals = msolver.eigs(A, M=M, n=5)
        
    es[i, :] = 1/vals.real
    NDof[i] = NN

print("eigens:\n", es)
print("NDof:\n", NDof)




