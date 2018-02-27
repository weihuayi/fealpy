import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from scipy.sparse import spdiags
from timeit import default_timer as timer

from petsc4py import PETSc

def linear_solver(dmodel, uh, dirichlet=None):
    start = timer()
    A = dmodel.get_left_matrix()
    b = dmodel.get_right_vector()
    end = timer()
    print("Construct linear system time:", end - start)
    if dirichlet is not None:
        AD, b = dirichlet.apply(A, b)

    start = timer()
    PA = PETSc.Mat().createAIJ(
            size=AD.shape, 
            csr=(AD.indptr, AD.indices,  AD.data)
            ) 
    Pb = PETSc.Vec().createWithArray(b)
    x = PETSc.Vec().createWithArray(uh)
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(PA)
    ksp.setFromOptions()
    ksp.solve(Pb, x)
    end = timer()
    print("Solve time:", end-start)
