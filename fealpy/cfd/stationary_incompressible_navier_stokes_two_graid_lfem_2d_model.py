from fealpy.backend import backend_manager as bm
from ..decorator import variantmethod
from ..model import ComputationalModel
from ..fem import DirichletBC
from .simulation.fem.stationary_incompressible_ns import BackwardEuler
from .simulation.fem.stationary_incompressible_ns import Newton
from .simulation.fem.stationary_incompressible_ns import Ossen
from .equation.stationary_incompressible_ns import StationaryIncompressibleNS

class StationaryIncompressibleNSLFEM2DModel(ComputationalModel):
    def __init__(self, pde):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pde = pde
        self.equation = StationaryIncompressibleNS(pde)
        self.fem = self.method()
        self.A = self.fem.BForm()
        self.F = self.fem.LForm()
        
    @variantmethod("Newton")
    def method(self): 
        self.fem = Newton(self.equation)
        self.method_str = "Newton"
        return self.fem
    
    @method.register("Ossen")
    def method(self): 
        self.fem = Ossen(self.equation)
        self.method_str = "Ossen"
        return self.fem
    
    @method.register("Backward_Euler")
    def method(self): 
        self.fem = BackwardEuler(self.equation)
        self.method_str = "Backward_Euler"
        return self.fem
    
    def update(self, u0):   
        self.fem.update(u0)
    
    def linear_system(self):
        A = self.A.assembly()
        F = self.F.assembly()
        return A, F


    @variantmethod('one_step')
    def run(self, maxit=10000, tol=1e-10):
        u0 = self.fem.uspace.function()
        p0 = self.fem.pspace.function()

        ugdof = self.fem.uspace.number_of_global_dofs()
        
        uerror1 = 0
        perror1 = 0 
        for i in range(maxit):
            self.update(u0)
            A, F = self.linear_system()
            
            BC = DirichletBC(
            (self.fem.uspace, self.fem.pspace), 
            gd=(self.pde.velocity, self.pde.pressure), 
            threshold=(self.pde.is_u_boundary, self.pde.is_p_boundary), 
            method='interp')
            A, F = BC.apply(A, F)
            x = self.solve(A, F)
            u0[:] = x[:ugdof]
            p0[:] = x[ugdof:]

            uerror0 = self.pde.mesh.error(self.pde.velocity, u0)
            perror0 = self.pde.mesh.error(self.pde.pressure, p0)
            res_u = bm.abs(uerror1 - uerror0)
            res_p = bm.abs(perror1 - perror0)
            if res_u + res_p < tol:
                print("Converged at iteration", i+1)
                break
            uerror1, perror1 = uerror0, perror0

        return u0, p0




    @variantmethod('direct')
    def solve(self, A, F, solver='scipy'):
        from fealpy.solver import spsolve
        return spsolve(A, F, solver = 'scipy')

    @solve.register('cg')
    def solve(self, A, F):
        pass
        
    @run.register('uniform_refine')
    def run(self):
        pass

    @run.register('bisect')
    def run(self):
        pass

    @variantmethod('error')
    def postprocess(self):
        pass




