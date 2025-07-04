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

    def simulation(self):
        return ossen_simulation(self)

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
        self.u_LSI = SourceIntegrator(q=q)
        L0.add_integrator(self.u_LSI) 
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
        @barycentric
        def u_LSI_coef(bcs, index):
            cbfcoef = cbf(bcs, index) if callable(cbf) else cbf
            result = cbfcoef 
            return result
        self.u_LSI.source = u_LSI_coef
       

class ossen_simulation(SimulationBase):
    def __init__(self, method):
        super().__init__(method)

    def _initialize_variables(self):
        pass

    def run_one_step(self, u0, p0):
        equation = self.equation
        pde = equation.pde
        uspace = self.method.uspace
        pspace = self.method.pspace
        ugdof = uspace.number_of_global_dofs()
        BC = DirichletBC(
            (uspace, pspace), 
            gd=(pde.velocity, pde.pressure), 
            threshold=(pde.is_u_boundary, pde.is_p_boundary), 
            method='interp'  
        )
        source = self.method.uspace.interpolate(pde.source)
        equation.set_coefs(body_force=source)
        self.BC = BC
        A = self.method.BForm()
        b = self.method.LForm()
        self.method.update(u0)
        A = A.assembly()
        b = b.assembly()
        A, b = BC.apply(A, b)
        x = spsolve(A, b, solver='mumps')
        u0[:] = x[:ugdof]
        p0[:] = x[ugdof:]

        return u0, p0

    def run(self, u0, p0):
        pde = self.equation.pde
        u0, p0 = self.run_one_step(u0, p0)
        uerror1 = 0
        perror1 = 0
        tol = self.method.tol
        for i  in range(self.method.max_iter):
            
            u0, p0 = self.run_one_step(u0, p0)
            uerror0 = pde.mesh.error(pde.velocity, u0)
            perror0 = pde.mesh.error(pde.pressure, p0)
            res_u = bm.abs(uerror1 - uerror0)
            res_p = bm.abs(perror1 - perror0)
            if res_u < tol and res_p < tol:
                print("Converged at iteration", i+1)
                break
            uerror1 = uerror0
            perror1 = perror0
        print("uerror",uerror0)
        print("perror",perror0)