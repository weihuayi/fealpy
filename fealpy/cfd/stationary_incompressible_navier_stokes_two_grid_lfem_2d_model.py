from fealpy.backend import backend_manager as bm
from ..decorator import variantmethod
from ..model import ComputationalModel
from ..fem import DirichletBC
from .simulation.fem.stationary_incompressible_ns import Stokes
from .simulation.fem.stationary_incompressible_ns import Newton
from .simulation.fem.stationary_incompressible_ns import Ossen
from .equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.utils import timer
import time

class StationaryIncompressibleNSTwoGridLFEM2DModel(ComputationalModel):
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
    
    @method.register("Stokes")
    def method(self): 
        self.fem = Stokes(self.equation)
        self.method_str = "Stokes"
        return self.fem
    
    def update(self, u0):   
        self.fem.update(u0)
    
    def linear_system(self):
        BForm = self.fem.BForm()
        LForm = self.fem.LForm()
        return BForm, LForm
    
    def lagrange_multiplier(self, A, b):
        from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
        from fealpy.sparse import COOTensor

        LagLinearForm = LinearForm(self.fem.pspace)
        LagLinearForm.add_integrator(SourceIntegrator(source=1))
        LagA = LagLinearForm.assembly()
        LagA = bm.concatenate([bm.zeros(self.fem.uspace.number_of_global_dofs()), LagA], axis=0)

        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))


        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b  = bm.concatenate([b, b0], axis=0)

        return A, b
    
    @variantmethod('one_step')
    def run(self, maxstep=1000, tol=1e-8):
        uh0 = self.fem.uspace.function()
        ph0 = self.fem.pspace.function()
        uh1 = self.fem.uspace.function()
        ph1 = self.fem.pspace.function()

        ugdof = self.fem.uspace.number_of_global_dofs()
        
        BForm, LForm = self.linear_system()
        for i in range(maxstep):
            self.logger.info(f"iteration: {i+1}")
            tmr = timer()
            next(tmr)
            start = time.time()
            self.update(uh0)
            A = BForm.assembly()
            tmr.send('迭代左端项矩阵组装时间')
            F = LForm.assembly()
            tmr.send('右端项组装时间')
            BC = DirichletBC(
                (self.fem.uspace, self.fem.pspace), 
                gd=(self.pde.velocity, self.pde.pressure), 
                threshold=(self.pde.is_velocity_boundary, self.pde.is_pressure_boundary), 
                method='interp')
            
            A, F = BC.apply(A, F)
            A, F = self.lagrange_multiplier(A ,F)
            tmr.send('边界处理时间')
            x = self.solve(A, F, solver='mumps')
            tmr.send('矩阵求解时间')
            #next(tmr)
            uh1[:] = x[:ugdof]
            ph1[:] = x[ugdof:-1]
            res_u = self.pde.mesh.error(uh0, uh1)
            res_p = self.pde.mesh.error(ph0, ph1)
            
            self.logger.info(f"res_u: {res_u}, res_p: {res_p}")
            uh0[:] = uh1
            ph0[:] = ph1
            uerror, perror = self.postprocess(uh1, ph1) 
            if res_u + res_p < tol:
                self.logger.info(f"Converged at iteration {i+1}")
                self.logger.info(f"res_u: {res_u}, res_p: {res_p}")
                break
        self.logger.info(f"final uerror: {uerror}, final perror: {perror}") 
        return uh1, ph1 

    @run.register('uniform_refine')
    def run(self, maxit = 5, maxstep = 1000, tol = 1e-8):
        for i in range(maxit):
            print('mesh', self.pde.mesh.number_of_cells())
            self.run['one_step'](maxstep, tol)
            self.pde.mesh.uniform_refine()
            self.equation = StationaryIncompressibleNS(self.pde)

    @variantmethod('direct')
    def solve(self, A, F, solver='scipy'):
        from fealpy.solver import spsolve
        return spsolve(A, F, solver = solver)

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    @solve.register('pcg')
    def solve(self, A, F):
        pass
        
    @variantmethod('error')
    def postprocess(self, uh, ph):
        uerror = self.pde.mesh.error(self.pde.velocity, uh)
        perror = self.pde.mesh.error(self.pde.pressure, ph)
        #print(f"uerror: {uerror}, perror: {perror}")
        return uerror, perror
