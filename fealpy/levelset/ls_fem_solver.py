import numpy as np

from ..fem import BilinearForm
from ..fem import LinearForm
from ..fem import ScalarConvectionIntegrator
from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ScalarMassIntegrator

from ..decorator import barycentric

from .ls_solver import LSSolver

from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import spsolve


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
        A = M + (dt/2) * C 
        b = M @ phi0 - (dt/2) * C @ phi0

        counter = IterationCounter(disp = False)
        phi0, info = lgmres(A, b, tol = tol, callback = counter)
        print("Convergence info:", info)
        print("Number of iteration of gmres:", counter.niter)
        return phi0

    def reinit(self, phi0, cellscale, dt = 0.0001, eps = 5e-6, nt = 4, alpha = None):
        """
        @brief 重新初始化水平集函数为符号距离函数
        """
        if alpha is None:
            alpha = 0.0625*cellscale

        space = self.space
        phi1 = space.function()
        phi1[:] = phi0
        phi2 = space.function()

        M = self.M

        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        S = bform.assembly()

        eold = 0   

        for _ in range(nt):

            @barycentric
            def f(bcs):
                grad = phi1.grad_value(bcs)
                val = 1 - np.sqrt(np.sum(grad**2, -1))
                val *= np.sign(phi0(bcs))
                return val
            
            LF = LinearForm(space)
            LF.add_domain_integrator( ScalarSourceIntegrator(f = f) )
            b0 = LF.assembly()
            b = M @ phi1 + dt * b0 - dt * alpha * (S @ phi1)

            phi2[:] = spsolve(M, b)
            error = space.mesh.error(phi2, phi1)
            print("重置:", error) 
            if eold < error and error< eps :
                break
            else:
                phi1[:] = phi2
                eold = error

        return phi1

