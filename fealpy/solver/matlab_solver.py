import numpy as np
from timeit import default_timer as timer

class MatlabSolver:
    def __init__(self, matlab):
        """
        Parameters
        ----------
        matlab : tranplant instance, 


        Example
        -------
        >> import transplant
        >> from fealpy.solver import MatlabSolver
        >> matalb = transplant.Matlab()
        >> solver = MatlabSolver(matlab)
        """
        self.matlab = matlab

    def divide(self, A, b):
        start = timer()
        x = self.matlab.mldivide(A, b.reshape(-1, 1))
        end = timer()
        return x.reshape(-1)

    def eigs(self, A, M=None, scale=None, n=1, eig_type='SM'):
        if scale is None:
            u, d, flag = self.matlab._call('eigs', [A, M, n, eig_type])
            return u.reshape(-1), d
        else:
            u, d, flag = self.matlab._call('eigs', [A + scale*M, M, n, eig_type])
            return u.reshape(-1), d - scale

    def mumps_solver(self, A, b):
        start = timer()
        u, rel = self.matlab._call('mumps_solver', [A, b])
        end = timer()
        print("The time of mumps direct solver:", end - start)
        print("The residual is:", rel)
        return u.reshape(-1)

    def ifem_amg_solver(self, A, b):
        start = timer()
        u = self.matlab._call('amg', [A, b])
        end = timer()
        print("The time of ifem amg solver:", end - start)
        return u.reshape(-1)

