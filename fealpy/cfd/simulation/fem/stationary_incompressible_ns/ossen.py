from .....backend import backend_manager as bm
from .....fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from .....fem import DirichletBC
from .....fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator,
                     ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, BoundaryFaceSourceIntegrator)
from .....decorator import barycentric
from .....fem import BoundaryPressWorkIntegrator
from .....solver import spsolve
from ..fem_base import FEM
from ...simulation_base import SimulationBase, SimulationParameters


class Ossen(FEM):
    """Ossen Interative Method""" 
    
    def __init__(self, equation, boundary_threshold=None):
        FEM.__init__(self, equation)
        self.threshold = boundary_threshold

    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        threshold = self.threshold
        
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
       
