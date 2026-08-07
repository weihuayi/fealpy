from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm
from fealpy.fem import (ScalarConvectionIntegrator, ScalarDiffusionIntegrator,
                     SourceIntegrator, ScalarMassIntegrator)
from fealpy.decorator import barycentric
from fealpy.functionspace import LagrangeFESpace

from ..iterative_method import IterativeMethod 


class StationaryTurbulentKineticEnergyPicard(IterativeMethod):
    """
    Picard iteration method for stationary turbulent kinetic energy equation.
    """

    def BForm(self):
        self.kspace = LagrangeFESpace(self.mesh, p=2)
        kspace = self.kspace
        q = 5

        A = BilinearForm(kspace)
        self.k_BC = ScalarConvectionIntegrator(q=q)
        self.k_BD = ScalarDiffusionIntegrator(q=q)
        self.k_BM = ScalarMassIntegrator(q=q)

        A.add_integrator(self.k_BC)
        A.add_integrator(self.k_BD)
        A.add_integrator(self.k_BM)

        return A
    
    def LForm(self):
        kspace = self.kspace
        q = 5

        L = LinearForm(kspace)
        self.k_LP = SourceIntegrator(q=q)
        L.add_integrator(self.k_LP)
        
        return L
    
    def update(self, u1, k0, omega0, mu_t):
        equation = self.equation
        cc = equation.coef_convection
        cd = equation.coef_diffusion
        cr = equation.coef_reaction
        cp = equation.coef_production
        
        ## BilinearForm
        @barycentric
        def k_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u1(bcs, index)
        self.k_BC.coef = k_BC_coef

        @barycentric
        def k_BD_coef(bcs, index):
            cdcoef = cd(bcs, index)[..., bm.newaxis] if callable(cd) else cd
            cdcoef += equation.pde.sigma_k * mu_t
            return cdcoef
        self.k_BD.coef = k_BD_coef

        @barycentric
        def k_BM_coef(bcs, index):
            crcoef = cr(bcs, index)[..., bm.newaxis] if callable(cr) else cr
            return crcoef * omega0(bcs, index)
        self.k_BM.coef = k_BM_coef

        ## LinearForm
        @barycentric
        def k_LP_coef(bcs, index):
            result = equation.pde.production_k(u0 = u1, 
                                               k0 = k0, 
                                               omega0 = omega0, 
                                               mu_t = mu_t, 
                                               bcs = bcs, 
                                               index = index)
            return result
        self.k_LP.source = k_LP_coef

