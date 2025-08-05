from ....backend import backend_manager as bm
from ....backend import TensorLike
from ....fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from ....fem import DirichletBC
from ....fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator,
                     ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, BoundaryFaceSourceIntegrator)
from ....decorator import barycentric
from ....fem import BoundaryPressWorkIntegrator

from .fem_base import FEM
from ..simulation_base import SimulationBase, SimulationParameters


class Ossen(FEM):
    """Ossen Interative Method""" 
    def __init__(self, equation, mesh):
        super().__init__(equation, mesh)
        
        if self.equation.pressure_neumann == False:
            self.threshold = self.equation.pde.is_pressure_boundary
    
    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        threshold = self.threshold
        
        A00 = BilinearForm(uspace)
        self.u_BM = ScalarMassIntegrator(q=q)
        self.u_BC = ScalarConvectionIntegrator(q=q)
        self.u_BVW = ViscousWorkIntegrator(q=q)
        self.u_BFBF = FluidBoundaryFrictionIntegrator(q=q, threshold=threshold)
        A00.add_integrator(self.u_BM)
        A00.add_integrator(self.u_BC)
        A00.add_integrator(self.u_BVW)
        A00.add_integrator(self.u_BFBF)
        
        A01 = BilinearForm((pspace, uspace))
        self.u_BPW = PressWorkIntegrator(q=q)
        self.u_BBPW = BoundaryPressWorkIntegrator(q=q, threshold=threshold)
        A01.add_integrator(self.u_BPW)
        A01.add_integrator(self.u_BBPW)

        A10 = BilinearForm((pspace, uspace))
        self.p_BPW = PressWorkIntegrator(q=q)
        A10.add_integrator(self.p_BPW)
        
        A = BlockForm([[A00, A01], [A10.T, None]]) 
        return A
        
    def LForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q

        L0 = LinearForm(uspace)
        self.u_LSI = SourceIntegrator(q=q)
        L0.add_integrator(self.u_LSI) 
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def update(self, u0, u00): 
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative 
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BM.coef = ctd/dt
        self.u_BVW.coef = 2*cv
        self.u_BPW.coef = -pc
        self.p_BPW.coef = 1
        self.u_BFBF.coef = -cv
        self.u_BBPW.coef = pc 

        @barycentric
        def u_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            return cccoef * u0(bcs, index)
        self.u_BC.coef = u_BC_coef

        ## LinearForm 
        @barycentric
        def u_LSI_coef(bcs, index):
            ctdcoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            cbfcoef = cbf(bcs, index) if callable(cbf) else cbf
            
            result = ctdcoef * u00(bcs, index) / dt
            result += cbfcoef 
            return result
        self.u_LSI.source = u_LSI_coef
       

