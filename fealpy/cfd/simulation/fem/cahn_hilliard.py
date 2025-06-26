from fealpy.backend import backend_manager as bm
from ....fem import (ScalarMassIntegrator, ScalarConvectionIntegrator,PressWorkIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, GradSourceIntegrator, 
                     ScalarDiffusionIntegrator)
from ....fem import (BilinearForm, LinearForm, BlockForm, LinearBlockForm)
from ....decorator import barycentric



class CahnHilliardModel:
    def __init__(self, equation, space):
        self.space = space
        self.equation = equation
        self.s = 1  #稳定化参数
        self.q = 5

    def BForm(self):
        phispace = self.space
        q = self.q

        A00 = BilinearForm(phispace)
        self.BM_phi = ScalarMassIntegrator(q=q)
        self.BC_phi = ScalarConvectionIntegrator(q=q) 
        A00.add_integrator(self.BM_phi)
        A00.add_integrator(self.BC_phi)

        A01 = BilinearForm(phispace)
        self.BD_phi = ScalarDiffusionIntegrator(q=q)
        A01.add_integrator(self.BD_phi)
        
        A10 = BilinearForm(phispace)
        self.BD_mu = ScalarDiffusionIntegrator(q=q)
        self.BM_mu0 = ScalarMassIntegrator(q=q)
        A10.add_integrator(self.BD_mu)
        A10.add_integrator(self.BM_mu0)

        A11 = BilinearForm(phispace)
        self.BM_mu1 = ScalarMassIntegrator(q=q)
        A11.add_integrator(self.BM_mu1)  

        A = BlockForm([[A00, A01], [A10, A11]]) 
        return A

    def LForm(self):
        phispace = self.space
        q = self.q

        L0 = LinearForm(phispace)
        self.LS_phi = SourceIntegrator(q=q)
        L0.add_integrator(self.LS_phi)

        L1 = LinearForm(phispace)
        self.LS_mu = SourceIntegrator(q=q)
        L1.add_integrator(self.LS_mu)

        L = LinearBlockForm([L0, L1])
        return L

    def update(self, u_0, u_1, phi_0, phi_1):
        dt = self.dt
        s = self.s

        cm = self.equation.coef_mobility
        cf = self.equation.coef_free_energy
        ci = self.equation.coef_interface
        
        #BilinearForm
        self.BM_phi.coef = 3/(2*dt)
        self.BC_phi.coef = 2*u_1

        self.BD_phi.coef = cm
        
        self.BD_mu.coef = -ci
        self.BM_mu0.coef = -s * cf

        self.BM_mu1.coef = 1
        
        #LinearForm
        @barycentric
        def LS_phi_coef(bcs, index):
            result = (4*phi_1(bcs, index) - phi_0(bcs, index))/(2*dt)
            result += bm.einsum('jid, jid->ji', u_0(bcs, index), phi_1.grad_value(bcs, index))
            return result

        self.LS_phi.source = LS_phi_coef

        @barycentric
        def LS_mu_coef(bcs, index): 
            result = -2*(1+s)*phi_1(bcs, index) + (1+s)*phi_0(bcs, index)
            result += 2*phi_1(bcs, index)**3 - phi_0(bcs, index)**3
            result *= cf
            return result
        self.LS_mu.source = LS_mu_coef
