
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, block_diag
from scipy.sparse import spdiags, eye, bmat, tril, triu
from scipy.sparse.linalg import cg, inv, dsolve,  gmres, LinearOperator, spsolve_triangular
import pyamg


class SaddlePointFastSolver():

    def __init__(self, A, B, D, F):
        """

        Notes
        -----

        [[A, B]
         [B, 0]] 
        """
        self.A = A
        self.B = B
        self.D = D

    def linear_operator(self):
        pass

    def solve(self):
        pass
