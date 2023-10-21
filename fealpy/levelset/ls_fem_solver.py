import numpy as np

from ..fem import BilinearForm
from ..fem import ScalarConvectionIntegrator
from ..fem import ScalarMassIntegrator
from scipy.sparse.linalg import lgmres

from .ls_solver import LSSolver

class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))

class LSFEMSolver(LSSolver):
    def __init__(self, space, u=None):
        self.space = space
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarMassIntegrator())
        self.M = bform.assembly() # TODO: 实现快速组装方法

        self.u = u
        if u is not None:
            bform = BilinearForm(space)
            bform.add_domain_integrator(ScalarConvectionIntegrator(c = u))
            self.C = bform.assembly() # TODO：实现快速组装方法

    def solve(self, phi0, dt, u=None, tol=1e-8):
        space = self.space
        M = self.M

        if u is None:
            C = self.C 
            if C is None:
                raise ValueError(" Velocity `u` is None! You must offer velocity!")
        else:
            bform = BilinearForm(space)
            bform.add_domain_integrator(ScalarConvectionIntegrator(c = u))
            C = bform.assembly()
        A = M + C 
        b = M@phi0 - C@phi0

        x, info = lgmres(self.A, b, tol=tol, callback=counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)
        return x

    def reinit(self, phi0):
        """
        @brief 重新初始化水平集函数为符号距离函数
        """
        pass
