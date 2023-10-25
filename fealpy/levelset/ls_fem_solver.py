import numpy as np

from ..fem import BilinearForm
from ..fem import LinearForm
from ..fem import ScalarConvectionIntegrator
from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ScalarMassIntegrator

from ..decorator import barycentric

from .ls_solver import LSSolver

from scipy.sparse.linalg import spsolve


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

        phi0 = self.solve_system(A, b, tol = tol)

        return phi0

    def reinit(self, phi0, dt = 0.0001, eps = 5e-6, nt = 4, alpha = None):
        '''
        TODO wrong!
        '''
        space = self.space
        mesh = space.mesh

        cellscale = np.max(mesh.entity_measure('cell'))
        if alpha is None:
            alpha = 0.0625*cellscale

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
            def f(bcs, index):
                grad = phi1.grad_value(bcs)
                val = 1 - np.sqrt(np.sum(grad**2, -1))
                val *= np.sign(phi0(bcs))
                return val
            
            lform = LinearForm(space)
            lform.add_domain_integrator( ScalarSourceIntegrator(f = f) )
            b0 = lform.assembly()
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

