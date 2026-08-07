from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import (ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator)
from fealpy.decorator import barycentric

from ..iterative_method import IterativeMethod 


class Ossen(IterativeMethod):
    """Ossen Iterative Method""" 
    
    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        
        A00 = BilinearForm(uspace)
        self.u_BC = ScalarConvectionIntegrator(q=q)
        self.u_BVW = ViscousWorkIntegrator(q=q)
        
        A00.add_integrator(self.u_BC)
        A00.add_integrator(self.u_BVW)

        A01 = BilinearForm((pspace, uspace))
        self.u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(self.u_BPW)
        
        A = BlockForm([[A00, A01], [A01.T, None]]) 
        return A
        
    def LForm(self):        
        pspace = self.pspace
        uspace = self.uspace
        q = self.q

        L0 = LinearForm(uspace)
        self.u_LSI = SourceIntegrator(q=q)
        self.u_source_LSI = SourceIntegrator(q=q)
        L0.add_integrator(self.u_LSI)
        L0.add_integrator(self.u_source_LSI) 
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def update(self, u0, k0, omega0): 
        equation = self.equation
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BPW.coef = -pc

        @barycentric
        def u_BVM_coef(bcs, index):
            points = self.uspace.mesh.bc_to_point(bcs, index)
            mu_t = equation.pde.tur_mu(u0=u0, k0=k0, omega0=omega0, bcs=bcs, points= points)
            mu_t = bm.minimum(mu_t, 1e-4)
            self.mu_t = mu_t
            cvcoef = cv(bcs, index)[..., bm.newaxis] if callable(cv) else cv
            cvcoef += mu_t
            return cvcoef
        self.u_BVW.coef = u_BVM_coef

        @barycentric
        def u_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            cccoef *= u0(bcs, index)
            return cccoef
        self.u_BC.coef = u_BC_coef

        ## LinearForm 
        @barycentric
        def u_LSI_coef(bcs, index):
            scoef = -2/3 * self.equation.pde.rho
            scoef *= k0.grad_value(bcs, index)
            return scoef
        self.u_LSI.source = u_LSI_coef
        self.u_source_LSI.source = cbf
       
