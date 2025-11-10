from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.fem import DirichletBC
from fealpy.cfd.equation import IncompressibleNS, CahnHilliard
from .simulation.time import UniformTimeLine


class CHNSFEMModel(ComputationalModel):
    def __init__(self, pde, mesh=None, options = None):
        super().__init__(pbar_log = True, log_level = "INFO")
        self.pde = pde
        self.ns_equation = IncompressibleNS(self.pde, init_variables=False)
        self.ch_equation = CahnHilliard(self.pde, init_variables=False)
        self.timeline = UniformTimeLine(0, 1, 100)
        self.options = options
        
        if mesh is None:
            if hasattr(pde, 'mesh'):
                self.mesh = pde.mesh
            else:
                raise ValueError("Not found mesh!")
        else:
            self.mesh = mesh
        
        self.ns_fem = self.ns_method()
        self.ch_fem = self.ch_method()

        if options is not None:
            self.solve.set(options['solve'])
            self.ns_fem = self.ns_method[options['ns_method']]()
            self.ch_fem = self.ch_method[options['ch_method']]()
            self.run.set(options['run'])
            self.maxit = options.get('maxit', 5)
            self.maxstep = options.get('maxstep', 1)
            self.tol = options.get('tol', 1e-10)

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the computational model configuration."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  ns_equation    : {self.ns_equation.__class__.__name__}\n"
        s += f"  ch_equation    : {self.ch_equation.__class__.__name__}\n"
        s += f"  pde            : {self.pde.__class__.__name__}\n"
        s += f"  ns_method      : {self.ns_method_str}\n"
        s += f"  ch_method      : {self.ch_method_str}\n"
        s += f"  run            : {self.run_str}\n"
        s += f"  maxsteps       : {self.maxstep}\n"
        s += f"  tol            : {self.tol}\n"
        s += f"  solve          : {self.solve_str}\n"
        s += ")"
        self.logger.info(f"\n{s}")
   

    def set_timeline(self, timeline: UniformTimeLine):
        self.timeline = timeline

    @variantmethod('BDF2')
    def ns_method(self):
        from .simulation.fem.incompressible_ns import BDF2
        self.ns_fem = BDF2(self.ns_equation, self.mesh)
        self.ns_method_str = "BDF2"
        return self.ns_fem

    @variantmethod('ch_fem')
    def ch_method(self):
        from .simulation.fem import CahnHilliardFEM
        phispace = self.ns_fem.uspace.scalar_space
        self.ch_fem = CahnHilliardFEM(self.ch_equation, phispace)
        self.ch_method_str = " CahnHilliardFEM"
        return self.ch_fem

    def ns_update(self, u_0, u_1):
        self.ns_fem.update(u_0, u_1)

    def ch_update(self, u_0, u_1, phi_0, phi_1):
        self.ch_fem.update(u_0, u_1, phi_0, phi_1)

    @variantmethod('direct')
    def solve(self, A, F, solver = 'mumps'):
        """Solve the linear system Ax = F."""
        self.solve_str = "direct"
        from fealpy.solver import spsolve
        return spsolve(A, F, solver=solver)
    
    @solve.register('cg')
    def solve(self, A, b, x0 = None, 
              M = None, *,batch_first: bool=False,
              atol: float=1e-12, rtol: float=1e-8, maxit = 10000,
              returninfo: bool=False):
        """Solve the linear system Ax = F."""
        self.solve_str = "cg"
        from fealpy.solver import cg
        return cg(A, b, x0 = x0, M = M, batch_first=batch_first,
                  atol=atol, rtol=rtol, maxit = maxit, returninfo=returninfo)

    @variantmethod('one_step')
    def run(self, phi0, phi1, mu1, u0, u1, p1):
        import time      
        self.run_str = "one_step"
        pde = self.pde
        ns_fem = self.ns_fem
        ch_fem = self.ch_fem
        phispace = ch_fem.space
        phigdof = phispace.number_of_global_dofs()
        ugdof = ns_fem.uspace.number_of_global_dofs()
        pgdof = ns_fem.pspace.number_of_global_dofs()
        ch_BForm = ch_fem.BForm()
        ch_LForm = ch_fem.LForm()
        ns_BForm = ns_fem.BForm()
        ns_LForm = ns_fem.LForm()

        t0 = time.time()
        self.ch_update(u0, u1, phi0, phi1)
        ch_A = ch_BForm.assembly()
        ch_b = ch_LForm.assembly()
        t1 = time.time()
        ch_x = self.solve(ch_A, ch_b, 'mumps')
        t2 = time.time()

        phi2 = ch_x[:phigdof]
        mu2 = ch_x[phigdof:]  
        
        # 更新NS方程参数
        t3 = time.time()
        pde.rho = pde.rho_update(phi1) 
        
        self.ns_equation.set_coefficient('time_derivative', pde.rho)
        self.ns_equation.set_coefficient('convection', pde.rho)
        self.ns_equation.set_coefficient('body_force', pde.body_force)

        ns_fem.update(u0, u1)
        
        ns_A = ns_BForm.assembly()
        ns_b = ns_LForm.assembly()

        is_bd = ns_fem.uspace.is_boundary_dof((pde.is_ux_boundary, pde.is_uy_boundary), method='interp')
        is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
        gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
        BC = DirichletBC((ns_fem.uspace, ns_fem.pspace), gd=gd, threshold=is_bd, method='interp')
        
        ns_A,ns_b = BC.apply(ns_A, ns_b)
        t4 = time.time() 
        ns_x = self.solve(ns_A, ns_b, 'mumps')
        t5 = time.time()

        print("CH组装时间:", t1-t0)
        print("求解CH方程时间:", t2-t1)
        print("NS组装时间:", t4-t3)
        print("求解NS方程时间:", t5-t4)
        u2 = ns_x[:ugdof]
        p2 = ns_x[ugdof:]
            
        u0[:] = u1[:]
        u1[:] = u2[:]
        phi0[:] = phi1[:]
        phi1[:] = phi2[:]
        mu1[:] = mu2[:]
        p1[:] = p2[:]

        return phi0, phi1, mu1, u0, u1, p1

        
    