
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, block_diag
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg, inv, dsolve,  gmres, lgmres, LinearOperator, spsolve_triangular

from fealpy.decorator import timer


class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, rk))

class FastSover():
    def __init__(self, A, F):
        self.A = A
        self.F = F

    @timer
    def solve(self, tol=1e-12):
        A = self.A
        F = self.F
        counter = IterationCounter()
        x, info = lgmres(A, F, tol=1e-12, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)

        return x 

