from .....backend import backend_manager as bm
from .....backend import TensorLike
from .....fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from .....fem import DirichletBC
from .....fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator,
                     ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, BoundaryFaceSourceIntegrator)
from .....decorator import barycentric

from ..fem_base import FEM
from ...simulation_base import SimulationBase, SimulationParameters
from .....solver import spsolve


class Newton(FEM):
    """Newton Interative Method""" 
    
    def __init__(self, equation):
        FEM.__init__(self, equation)

    def simulation(self):
        return newton_simulation(self)

    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        
        A00 = BilinearForm(uspace)
        self.u_BM_netwon = ScalarMassIntegrator(q=q)
        self.u_BC = ScalarConvectionIntegrator(q=q)
        self.u_BVW = ScalarDiffusionIntegrator(q=q)
        #self.u_BVW = ViscousWorkIntegrator(q=q)
        A00.add_integrator(self.u_BM_netwon)
        A00.add_integrator(self.u_BC)
        A00.add_integrator(self.u_BVW)
        
        A01 = BilinearForm((pspace, uspace))
        self.u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(self.u_BPW)
       
        A11 = BilinearForm(pspace)
        A11.add_integrator(ScalarMassIntegrator(coef=1e-10))
        A = BlockForm([[A00, A01], [A01.T, A11]]) 
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
        self.u_BPW.coef = -1

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
    
    

  
