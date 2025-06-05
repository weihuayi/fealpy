from fealpy.backend import backend_manager as bm
from .fem_base import FEM
from ....fem import (ScalarMassIntegrator, ScalarConvectionIntegrator,PressWorkIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, GradSourceIntegrator, 
                     ScalarDiffusionIntegrator)
from ....fem import (BilinearForm, LinearForm, BlockForm, LinearBlockForm)
from fealpy.decorator import barycentric
from fealpy.fem import ScalarNeumannBCIntegrator


class BDF2(FEM):
    
    def __init__(self, equation):
        super().__init__(equation)
     
    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q


        A00 = BilinearForm(uspace)
        self.BM = ScalarMassIntegrator(q=q)
        self.BC = ScalarConvectionIntegrator(q=q)
        #self.BD = ScalarDiffusionIntegrator(q=q)
        self.BD = ViscousWorkIntegrator(q=q)
        A00.add_integrator(self.BM)
        A00.add_integrator(self.BC)
        A00.add_integrator(self.BD)

        A01 = BilinearForm((pspace, uspace))
        self.BPW0 = PressWorkIntegrator(q=q)
        A01.add_integrator(self.BPW0) 

        A10 = BilinearForm((pspace, uspace))
        self.BPW1 = PressWorkIntegrator(q=q)
        A10.add_integrator(self.BPW1)
        
        A = BlockForm([[A00, A01], [A10.T, None]]) 
        return A

    def LForm(self, q=None):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        q = self.q

        L0 = LinearForm(uspace) 
        self.LSI_U = SourceIntegrator(q=q)
        L0.add_integrator(self.LSI_U)

        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def update(self, u_0, u_1):
        dt = self.dt
        equation = self.equation
        ctd = equation.coef_time_derivative 
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.BM.coef = 3*ctd/(2*dt)
        def BC_coef(bcs, index): 
            ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result = 2* ccoef * u_1(bcs, index)
            return result
        self.BC.coef = BC_coef

        self.BD.coef = 2*cv 
        #self.BD.coef = cv 
        self.BPW0.coef = -pc
        self.BPW1.coef = -1

        ## LinearForm
        
        @barycentric
        def LSI_U_coef(bcs, index):
            masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result0 =  masscoef * (4*u_1(bcs, index) - u_0(bcs, index)) / (2*dt)
            
            ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result1 = ccoef*bm.einsum('cqij, cqj->cqi', u_1.grad_value(bcs, index), u_0(bcs, index))
            cbfcoef = cbf(bcs, index) if callable(cbf) else cbf
            
            result = result0 + result1 + cbfcoef
            return result

        self.LSI_U.source = LSI_U_coef
   
        
