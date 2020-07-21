import numpy as np
from timeit import default_timer as timer

from petsc4py import PETSc

from ..decorator import timer


class PETScSolver():
    def __init__(self):
        pass

    @timer
    def solve(self, A, F, uh):
        PA = PETSc.Mat().createAIJ(
                size=A.shape, 
                csr=(A.indptr, A.indices,  A.data)
                ) 
        PF = PETSc.Vec().createWithArray(F)
        x = PETSc.Vec().createWithArray(uh)
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        # and incomplete Cholesky
        ksp.setType('cg')
        # and incomplete Cholesky
        ksp.getPC().setType('gamg')
        ksp.setOperators(PA)
        ksp.setFromOptions()
        ksp.solve(PF, x)
        return x

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

def minres(A, b, uh):

    PA = PETSc.Mat().createAIJ(
            size=A.shape, 
            csr=(A.indptr, A.indices,  A.data)
            ) 
    Pb = PETSc.Vec().createWithArray(b)
    x = PETSc.Vec().createWithArray(uh)

    ksp = PETSc.KSP().create()
    #ksp.setType('minres')
    pc = ksp.getPC()
    pc.setType('none')
    ksp.setOperators(PA)
    ksp.setFromOptions()
    ksp.solve(Pb, x)

