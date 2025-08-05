from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod,cartesian
from fealpy.model import ComputationalModel
from fealpy.fem import DirichletBC

from .equation import IncompressibleNS
from .simulation.time import UniformTimeLine


class IncompressibleNSLFEM2DModel(ComputationalModel):
    def __init__(self, pde, mesh=None, options = None):
        super().__init__(pbar_log = True, log_level = "INFO")
        self.pde = pde
        self.equation = IncompressibleNS(self.pde)
        self.timeline = UniformTimeLine(0, 1, 100)
        
        if mesh is None:
            if hasattr(pde, 'mesh'):
                self.mesh = pde.mesh
            else:
                raise ValueError("Not found mesh!")
        else:
            self.mesh = mesh
        
        self.fem = self.method()

        if options is not None:
            self.solve.set(options['solve'])
            self.fem = self.method[options['method']]()
            self.timeline = UniformTimeLine(options['T0'], options['T1'], options['nt'])

            run = self.run[options['run']]
            if options['run'] == 'uniform_refine':
                run(maxit=options['maxit'], maxstep=options['maxstep'], tol=options['tol'], apply_bc=options['apply_bc'], postprocess=options.get('postprocess', 'error'))
            else:  # 'one_step' 或其他
                run(maxstep=options['maxstep'], tol=options['tol'], apply_bc=options['apply_bc'], postprocess=options.get('postprocess', 'error'))

    @variantmethod("Newton")
    def method(self):
        from .simulation.fem.incompressible_ns import Newton
        self.fem = Newton(self.equation, self.mesh)
        self.method_str = "Newton"
        return self.fem
    
    @method.register("Ossen")
    def method(self):
        from .simulation.fem import Ossen
        self.fem = Ossen(self.equation, self.mesh)
        self.method_str = "Ossen"
        return self.fem
    
    @method.register("IPCS")
    def method(self):
        from .simulation.fem.incompressible_ns import IPCS
        self.fem = IPCS(self.equation, self.mesh)
        self.method_str = "IPCS"
        return self.fem
    
    @method.register("BDF2")
    def method(self):
        from .simulation.fem import BDF2
        self.fem = BDF2(self.equation, self.mesh)
        self.method_str = "BDF2"
        return self.fem

    def update(self, u0, uk):
        self.fem.update(u0, uk)

    def linear_system(self):
        BForm = self.fem.BForm()
        LForm = self.fem.LForm()
        return BForm, LForm
    

    @variantmethod('direct')
    def solve(self, A, F, solver = 'mumps'):
        from fealpy.solver import spsolve
        return spsolve(A, F, solver=solver)

    @variantmethod('main')
    def run(self, maxstep = 10, tol = 1e-10, postprocess = 'error'):
        
        mesh = self.mesh         
        fem = self.fem
        pde = self.pde
        timeline = self.timeline
        fem.dt = timeline.dt

        u0 = fem.uspace.interpolate(cartesian(lambda p: pde.velocity(p, t = timeline.T0)))
        p0 = fem.pspace.interpolate(cartesian(lambda p: pde.pressure(p, t = timeline.T0)))
        
        u0 = fem.uspace.function()
        p0 = fem.pspace.function()
        for i in range(timeline.NL-1):
            t  = timeline.current_time()
            print("time=", t)
            
            u1,p1 = self.run['one_step'](u0, p0, maxstep, tol)

            u0[:] = u1
            p0[:] = p1

            uerror, perror = self.error(u0, p0, t= timeline.next_time()) 
            timeline.advance()
        uerror, perror = self.error(u0, p0, t= timeline.T1)  
        return u0, p0
    
    @run.register('one_step')
    def run(self, u0, p0, maxstep=10, tol=1e-12):
        fem = self.fem
        pde = self.pde

        if self.method_str == "IPCS":
            BCu = DirichletBC(space=fem.uspace, 
                gd = cartesian(lambda p : pde.velocity_dirichlet(p, self.timeline.next_time())), 
                threshold=pde.is_velocity_boundary, 
                method='interp')

            BCp = DirichletBC(space=fem.pspace, 
                gd = cartesian(lambda p : pde.pressure_dirichlet(p, self.timeline.next_time())), 
                threshold=pde.is_pressure_boundary, 
                method='interp')
            
            uh1 = u0.space.function()
            uhs = u0.space.function()
            ph1 = p0.space.function()
            
            
            self.equation.set_coefficient('body_force', cartesian(lambda p: pde.source(p, self.timeline.next_time())))  
            
            A0, b0 = self.fem.predict_velocity(u0, p0, BC=BCu, return_form=False)
            uhs[:] = self.solve(A0, b0)

            A1, b1 = self.fem.pressure(uhs, p0, BC=BCp, return_form=False)
            if self.equation.pressure_neumann == True:
                ph1[:] = self.solve(A1, b1)[:-1]
            else:
                ph1[:] = self.solve(A1, b1)

            A2, b2 = self.fem.correct_velocity(uhs, p0, ph1, return_form=False)
            uh1[:] = self.solve(A2, b2)
            return uh1, ph1
        else:
            
            BForm, LForm = self.linear_system()
            ugdof = fem.uspace.number_of_global_dofs()
            
            uk0 = u0.space.function()
            uk1 = u0.space.function()
            uk0[:] = u0
            pk = p0.space.function()

            for j in range(maxstep): 
                self.equation.set_coefficient('body_force', cartesian(lambda p: pde.source(p, self.timeline.next_time())))  
                fem.update(uk0, u0)
                
                A = BForm.assembly()
                b = LForm.assembly()
                A, b = self.fem.apply_bc(A, b, self.pde, t=self.timeline.next_time())
                A, b = self.fem.lagrange_multiplier(A,b, 0)
            
                x = self.solve(A, b, 'mumps')
                uk1[:] = x[:ugdof]
                pk[:] = x[ugdof:-1]
                
                res_u = self.mesh.error(uk0, uk1)
                print(res_u)
                if res_u < tol:
                    break
                uk0[:] = uk1
            return uk1, pk
    

    @run.register('uniform_refine')
    def run(self, maxit=5, t0 = 0, nt = 300, maxstep = 10, tol = 1e-12, apply_bc = 'dirichlet', postprocess = 'error'):
        fem = self.fem
        for i in range(maxit):
            self.logger.info(f'mesh: {self.pde.mesh.number_of_cells()}')
            self.run['time_step'](t0, nt, maxstep, tol,apply_bc, postprocess)
            self.pde.mesh.uniform_refine()
            self.equation = IncompressibleNS(self.pde)

        
    @variantmethod('L2')
    def error(self, uh, ph, t):
        uerror = self.mesh.error(cartesian(lambda p : self.pde.velocity(p, t = t)), uh)
        perror = self.mesh.error(cartesian(lambda p : self.pde.pressure(p, t = t)), ph)
        self.logger.info(f"uerror: {uerror}, perror: {perror}")
        return uerror, perror
