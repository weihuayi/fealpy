from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import DirichletBC
from fealpy.fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator,
                     ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                     ViscousWorkIntegrator, SourceIntegrator, BoundaryFaceSourceIntegrator)
from fealpy.decorator import barycentric
from fealpy.fem import BoundaryPressWorkIntegrator

from ..fem_base import FEM


class Newton(FEM):
    """Newton Interative Method""" 
    
    def __init__(self, equation, boundary_threshold=None):
        FEM.__init__(self, equation)
        self.threshold = boundary_threshold

    def BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        q = self.q
        threshold = self.threshold
        
        A00 = BilinearForm(uspace)
        self.u_BM = ScalarMassIntegrator(q=q)
        self.u_BM_netwon = ScalarMassIntegrator(q=q)
        self.u_BC = ScalarConvectionIntegrator(q=q)
        #self.u_BF = FluidBoundaryFrictionIntegrator(q=q, threshold=threshold) 
        
        if self.equation.constitutive.value == 1:
            self.u_BVW = ScalarDiffusionIntegrator(q=q)
        elif self.equation.constitutive.value == 2:
            self.u_BVW = ViscousWorkIntegrator(q=q)
        else:
            raise ValueError(f"未知的粘性模型")

        A00.add_integrator(self.u_BM)
        A00.add_integrator(self.u_BM_netwon)
        A00.add_integrator(self.u_BC)
        A00.add_integrator(self.u_BVW)
        #A00.add_integrator(self.u_BF)
        
        
        A01 = BilinearForm((pspace, uspace))
        self.u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(self.u_BPW)

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
        self.u_LSI_f = SourceIntegrator(q=q)
        L0.add_integrator(self.u_LSI) 
        L0.add_integrator(self.u_LSI_f)

        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def update(self, uk, u0): 
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative 
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BM.coef = ctd/dt
        self.u_BVW.coef = cv
        self.u_BPW.coef = -pc
        self.p_BPW.coef = 1

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
            ctdcoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result = ctdcoef * uk(bcs, index) / dt
            result += cccoef*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            return result
        self.u_LSI.source = u_LSI_coef
        self.u_LSI_f.source = cbf 

