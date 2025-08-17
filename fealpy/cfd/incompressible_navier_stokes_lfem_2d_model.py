from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod,cartesian
from fealpy.model import ComputationalModel
from fealpy.fem import DirichletBC
from .equation import IncompressibleNS
from .simulation.time import UniformTimeLine
from fealpy.utils import timer


class IncompressibleNSLFEM2DModel(ComputationalModel):
    def __init__(self, pde, mesh=None, options = None):
        super().__init__(pbar_log = True, log_level = "INFO")
        self.pde = pde
        self.equation = IncompressibleNS(self.pde)
        self.T0, self.T1, self.nt = 0, 1, 100
        self.timeline = UniformTimeLine(self.T0, self.T1, self.nt)
        self.options = options
        
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
            self.T0, self.T1, self.nt =options['T0'], options['T1'], options['nt']
            self.timeline = UniformTimeLine(self.T0, self.T1, self.nt)
            self.run.set(options['run'])
            self.maxit = options.get('maxit', 5)
            self.maxstep = options.get('maxstep', 10)
            self.tol = options.get('tol', 1e-10)
            # self.apply_bc = self.apply_bc[options['apply_bc']]

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the computational model configuration."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  equation       : {self.equation.__class__.__name__}\n"
        s += f"  pde            : {self.pde.__class__.__name__}\n"
        s += f"  method         : {self.method_str}\n"
        s += f"  run            : {self.run_str}\n"
        s += f"  maxsteps       : {self.maxstep}\n"
        s += f"  tol            : {self.tol}\n"
        s += f"  solve          : {self.solve_str}\n"
        s += f"  error          : {self.error_str}\n"
        if self.options.get("run") == "uniform_refine":
            s += f"  maxit          : {self.maxit}\n"
        s += ")"
        self.logger.info(f"\n{s}")

    @variantmethod("Newton")
    def method(self):
        from .simulation.fem.incompressible_ns import Newton
        self.fem = Newton(self.equation, self.mesh)
        self.method_str = "Newton"
        return self.fem
    
    @method.register("Ossen")
    def method(self):
        from .simulation.fem.incompressible_ns.ossen import Ossen
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
        """Solve the linear system Ax = F."""
        self.solve_str = "direct"
        from fealpy.solver import spsolve
        return spsolve(A, F, solver=solver)

    @variantmethod('main')
    def run(self, maxstep = 10, tol = 1e-10):
        self.run_str = "main"
        mesh = self.mesh         
        pde = self.pde
        fem = self.fem
        fem.dt = self.timeline.dt
        maxstep = self.maxstep if self.options is not None else maxstep
        tol = self.tol if self.options is not None else tol

        u0 = fem.uspace.interpolate(cartesian(lambda p: pde.velocity(p, t = self.timeline.T0)))
        p0 = fem.pspace.interpolate(cartesian(lambda p: pde.pressure(p, t = self.timeline.T0)))
        
        for i in range(self.timeline.NL-1):
            t  = self.timeline.current_time()
            self.logger.info(f"time={t}")
            
            u1,p1 = self.run['one_step'](u0, p0, maxstep, tol)

            u0[:] = u1
            p0[:] = p1

            uerror, perror = self.error(u0, p0, t= self.timeline.next_time()) 
            self.timeline.advance()
        uerror, perror = self.error(u0, p0, t= self.timeline.T1)  
        return u0, p0
    
    @run.register('one_step')
    def run(self, u0, p0, maxstep=10, tol=1e-12):
        self.run_str = "one_step"
        fem = self.fem
        pde = self.pde
        maxstep = self.maxstep if self.options is not None else maxstep
        tol = self.tol if self.options is not None else tol

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
                # tmr = timer()
                # next(tmr)
                self.equation.set_coefficient('body_force', cartesian(lambda p: pde.source(p, self.timeline.next_time())))  
                fem.update(uk0, u0)
                
                A = BForm.assembly()
                # tmr.send('左端项组装时间')
                b = LForm.assembly()
                # tmr.send('右端项组装时间')
                A, b = self.fem.apply_bc(A, b, self.pde, t=self.timeline.next_time())
                # tmr.send('边界条件处理时间')
                A, b = self.fem.lagrange_multiplier(A, b, 0)
                # tmr.send('拉格朗日乘子处理时间')
            
                x = self.solve(A, b, 'mumps')
                # tmr.send('求解线性方程组时间')
                # next(tmr)
                uk1[:] = x[:ugdof]
                pk[:] = x[ugdof:-1]
                
                res_u = self.mesh.error(uk0, uk1)
                # print(res_u)
                if res_u < tol:
                    break
                uk0[:] = uk1
            return uk1, pk
    

    @run.register('uniform_refine')
    def run(self, maxit=5, maxstep = 10, tol = 1e-12, apply_bc = 'dirichlet', postprocess = 'error'): 
        self.run_str = "uniform_refine"       
        maxit = self.maxit if self.options is not None else maxit
        maxstep = self.maxstep if self.options is not None else maxstep
        tol = self.tol if self.options is not None else tol
        u_errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
        p_errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
        for i in range(maxit):
            self.logger.info(f'mesh: {self.pde.mesh.number_of_cells()}')
            self.timeline = UniformTimeLine(self.T0, self.T1, self.nt)
            uh,ph = self.run['main']( maxstep, tol)
            self.nt = self.nt*4
            self.equation = IncompressibleNS(self.pde)
            uerror, perror = self.error(uh, ph, t= self.T1)
            u_errorMatrix[0, i] = uerror
            p_errorMatrix[0, i] = perror
            order_u = bm.log2(u_errorMatrix[0,:-1]/u_errorMatrix[0,1:])
            order_p = bm.log2(p_errorMatrix[0,:-1]/p_errorMatrix[0,1:])
            self.pde.mesh.uniform_refine()
        self.logger.info(f"速度最终误差:" + ",".join(f"{uerror:.15e}" for uerror in u_errorMatrix[0,]))
        self.logger.info(f"order_u: " + ", ".join(f"{order_u:.15e}" for order_u in order_u))
        self.logger.info(f"压力最终误差:" + ",".join(f"{perror:.15e}" for perror in p_errorMatrix[0,]))  
        self.logger.info(f"order_p: " + ", ".join(f"{order_p:.15e}" for order_p in order_p))
        
    @variantmethod('L2')
    def error(self, uh, ph, t):
        """Compute the error between numerical solution and exact solution."""
        self.error_str = "L2"
        uerror = self.mesh.error(cartesian(lambda p : self.pde.velocity(p, t = t)), uh)
        perror = self.mesh.error(cartesian(lambda p : self.pde.pressure(p, t = t)), ph)
        self.logger.info(f"uerror: {uerror}, perror: {perror}")
        return uerror, perror
