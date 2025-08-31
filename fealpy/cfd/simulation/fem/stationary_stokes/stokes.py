from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ( PressWorkIntegrator, ScalarDiffusionIntegrator, SourceIntegrator)

from ..iterative_method import IterativeMethod 

class Stokes(IterativeMethod):
    """Newton Interative Method""" 
    
    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        
        A00 = BilinearForm(uspace)
        self.u_BVW = ScalarDiffusionIntegrator(q=q)
        
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
        self.u_source_LSI = SourceIntegrator(q=q)
        L0.add_integrator(self.u_source_LSI)
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def update(self): 
        equation = self.equation
        cv = equation.coef_viscosity
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BVW.coef = cv
        self.u_BPW.coef = -pc

        ## LinearForm 
        self.u_source_LSI.source = cbf
    
    

  

