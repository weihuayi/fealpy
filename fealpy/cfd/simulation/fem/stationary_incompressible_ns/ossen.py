from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import DirichletBC
from fealpy.fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator,
                     ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, BoundaryFaceSourceIntegrator)
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
        self.u_BVW = ScalarDiffusionIntegrator(q=q)
        A00.add_integrator(self.u_BC)
        A00.add_integrator(self.u_BVW)

        
        A01 = BilinearForm((pspace, uspace))
        self.u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(self.u_BPW)

        A10 = BilinearForm((pspace, uspace))
        self.p_BPW = PressWorkIntegrator(q=q)
        A10.add_integrator(self.p_BPW)
        
        A11 = BilinearForm(pspace)
        A11.add_integrator(ScalarMassIntegrator(coef=1/1e10))
        A = BlockForm([[A00, A01], [A10.T, A11]]) 
        return A
        
    def LForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q

        L0 = LinearForm(uspace)
        self.u_source_LSI = SourceIntegrator(q=q)
        L0.add_integrator(self.u_source_LSI) 
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def update(self, u0): 
        equation = self.equation
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BVW.coef = cv
        self.u_BPW.coef = -pc
        self.p_BPW.coef = 1

        @barycentric
        def u_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u0(bcs, index)
        self.u_BC.coef = u_BC_coef

        ## LinearForm 
        self.u_source_LSI.source = cbf
       
