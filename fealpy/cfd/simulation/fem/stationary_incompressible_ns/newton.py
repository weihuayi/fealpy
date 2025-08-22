from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import (ScalarMassIntegrator,ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator)
from fealpy.decorator import barycentric

from ..iterative_method import IterativeMethod 

class Newton(IterativeMethod):
    """Newton Interative Method""" 
    
    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        
        A00 = BilinearForm(uspace)
        self.u_BM_netwon = ScalarMassIntegrator(q=q)
        self.u_BC = ScalarConvectionIntegrator(q=q)

        if self.equation.constitutive.value == 1:
            self.u_BVW = ScalarDiffusionIntegrator(q=q)
        elif self.equation.constitutive.value == 2:
            self.u_BVW = ViscousWorkIntegrator(q=q)
        else:
            raise ValueError(f"未知的粘性模型")
        
        A00.add_integrator(self.u_BM_netwon)
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

    def update(self, u0): 
        equation = self.equation
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BVW.coef = cv
        self.u_BPW.coef = -pc

        @barycentric
        def u_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u0(bcs, index)
        self.u_BC.coef = u_BC_coef

        @barycentric
        def u_BM_netwon_coef(bcs,index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u0.grad_value(bcs, index)
        self.u_BM_netwon.coef = u_BM_netwon_coef

        ## LinearForm 
        @barycentric
        def u_LSI_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result = cccoef*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            return result
        self.u_LSI.source = u_LSI_coef
        self.u_source_LSI.source = cbf
    
    

  
