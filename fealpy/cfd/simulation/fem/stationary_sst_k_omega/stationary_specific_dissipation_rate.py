from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm
from fealpy.fem import (ScalarConvectionIntegrator, ScalarDiffusionIntegrator,
                     SourceIntegrator, ScalarMassIntegrator)
from fealpy.decorator import barycentric
from fealpy.functionspace import LagrangeFESpace

from ..iterative_method import IterativeMethod 

class StationarySpecificDissipationRatePicard(IterativeMethod):
    """Stationary Specific Dissipation Rate Picard Iterative Method"""
    def BForm(self):
        self.omegaspace = LagrangeFESpace(self.mesh, p=2)
        omegasapce = self.omegaspace
        q = 5

        A = BilinearForm(omegasapce)
        self.omega_BC = ScalarConvectionIntegrator(q=q)
        self.omega_BD = ScalarDiffusionIntegrator(q=q)
        self.omega_BM = ScalarMassIntegrator(q=q)
        self.omega_BCD = ScalarConvectionIntegrator(q=q)

        A.add_integrator(self.omega_BC)
        A.add_integrator(self.omega_BD)   
        A.add_integrator(self.omega_BM)
        A.add_integrator(self.omega_BCD)

        return A

    def LForm(self):
        omegasapce = self.omegaspace
        q = 5

        L = LinearForm(omegasapce)
        self.omega_LP = SourceIntegrator(q=q)
        L.add_integrator(self.omega_LP)

        return L
    
    def update(self, u1, k1, omega0, mu_t):
        equation = self.equation 
        cc = equation.coef_convection
        cds = equation.coef_dissipation
        cd = equation.coef_diffusion
        ccd = equation.coef_cross_diffusion
        cp = equation.coef_production

        ## BilinearForm
        @barycentric
        def omega_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u1(bcs, index)
        self.omega_BC.coef = omega_BC_coef

        @barycentric
        def omega_BM_coef(bcs, index):
            cdscoef = cds(bcs, index)[..., bm.newaxis] if callable(cds) else cds
            return cdscoef * omega0(bcs, index)
        self.omega_BM.coef = omega_BM_coef

        @barycentric
        def omega_BD_coef(bcs, index):
            cdcoef = cd(bcs, index)[..., bm.newaxis] if callable(cd) else cd
            cdcoef = equation.pde.sigma_omega * mu_t
            return cdcoef
        self.omega_BD.coef = omega_BD_coef

        @barycentric
        def omega_BCD_coef(bcs, index):
            ccdcoef = ccd(bcs, index)[bm.newaxis, bm.newaxis] if callable(ccd) else ccd
            points = self.omegaspace.mesh.bc_to_point(bcs, index)
            F1 = equation.pde.cross_diffuison_f1(k1=k1, 
                                                 omega0=omega0, 
                                                 points=points, 
                                                 bcs=bcs, 
                                                 index=index)
            ccdcoef *= (1 - F1)
            reciprocal_omega0 = 1/omega0
            ccdcoef *= reciprocal_omega0(bcs, index)
            ccdcoef = ccdcoef[..., None] * k1.grad_value(bcs, index)
            return ccdcoef
        self.omega_BCD.coef = omega_BCD_coef

        ## LinearForm
        @barycentric
        def omega_LP_coef(bcs, index):
            result = cp(bcs, index)[bm.newaxis, bm.newaxis] if callable(cp) else cp
            result /= mu_t
            result *= equation.pde.production_omega(u0 = u1, 
                                               k0 = k1, 
                                               mu_t = mu_t, 
                                               bcs = bcs, 
                                               index = index)
            return result
        self.omega_LP.source = omega_LP_coef



