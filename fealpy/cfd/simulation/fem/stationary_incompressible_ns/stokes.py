from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric

from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import DirichletBC
from fealpy.fem import (ScalarMassIntegrator,
                     PressWorkIntegrator, ScalarDiffusionIntegrator,
                     SourceIntegrator)

from fealpy.solver import spsolve

from ..fem_base import FEM
from ...simulation_base import SimulationBase, SimulationParameters


class Stokes(FEM):
    """Stokes Interative Method""" 
    
    def __init__(self, equation):
        FEM.__init__(self, equation)

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
        self.p_BPW.coef = 1

        ## LinearForm 
        @barycentric
        def u_LSI_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result = -cccoef*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            return result
        self.u_LSI.source = u_LSI_coef
        self.u_source_LSI.source = cbf

