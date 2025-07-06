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
        BForm = self.fem.BForm()
        LForm = self.fem.LForm()
        return BForm, LForm


    @variantmethod('one_step')
    def run(self, maxit=10000, tol=1e-10):
        uh0 = self.fem.uspace.function()
        ph0 = self.fem.pspace.function()
        uh1 = self.fem.uspace.function()
        ph1 = self.fem.pspace.function()

        ugdof = self.fem.uspace.number_of_global_dofs()
        
        BForm, LForm = self.linear_system()
        for i in range(maxit):
            print(f"iteration: {i}")
            self.update(uh0)
            A = BForm.assembly()
            F = LForm.assembly()
            BC = DirichletBC(
                (self.fem.uspace, self.fem.pspace), 
                gd=(self.pde.velocity, self.pde.pressure), 
                threshold=(self.pde.is_u_boundary, self.pde.is_p_boundary), 
                method='interp')
            
            A, F = BC.apply(A, F)
            x = self.solve(A, F)
            uh1[:] = x[:ugdof]
            ph1[:] = x[ugdof:]
            res_u = self.pde.mesh.error(uh0, uh1)
            res_p = self.pde.mesh.error(ph0, ph1)
            print(f"res_u: {res_u}, res_p: {res_p}")
            if res_u + res_p < tol:
                print("Converged at iteration", i+1)
                break
            uh0[:] = uh1
            ph0[:] = ph1
        uerror, perror = self.postprocess(uh1, ph1)
        print(f"final uerror: {uerror}, final perror: {perror}")
        return uh1, ph1




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

    @variantmethod('error')
    def postprocess(self, uh, ph):
        uerror = self.pde.mesh.error(self.pde.velocity, uh)
        perror = self.pde.mesh.error(self.pde.pressure, ph)
        print(f"uerror: {uerror}, perror: {perror}")
        return uerror, perror

        




