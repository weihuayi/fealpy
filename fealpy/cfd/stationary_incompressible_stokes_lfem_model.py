from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.cfd.equation.stationary_stokes import StationaryStokes

class StationaryIncompressibleStokesLFEMModel(ComputationalModel):
    """
    StationaryIncompressibleNSLFEMModel: Stationary Incompressible Navier-Stokes Finite Element Solver

    This class implements a finite element solver for the stationary incompressible Navier-Stokes equations.
    It supports multiple linearization methods (Stokes, Oseen, Newton), boundary condition handling,
    linear system assembly, solution, and error evaluation.

    Parameters
    equation : StationaryIncompressibleNS
        The equation object encapsulating the PDE data, function spaces, and operators.
    options : dict
        Solver options, including:
            - 'solve': linear solver variant, e.g., 'direct', 'pcg'
            - 'method': FEM linearization method, e.g., 'Stokes', 'Ossen', 'Newton'
            - 'run': run method, e.g., 'one_step', 'uniform_refine'
            - 'maxstep': max nonlinear iteration steps
            - 'maxit': max mesh refinement iterations (for 'uniform_refine')
            - 'tol': convergence tolerance

    Attributes
    pde : object
        The PDE data including viscosity, velocity, pressure, etc.
    fem : object
        Finite element method object corresponding to the selected linearization method.
    solve : variant method
        Linear solver variant (e.g., direct, pcg, amg).
    run : variant method
        Execution strategy (e.g., one_step solve, adaptive or uniform mesh refinement).
    method : variant method
        Linearization method for the Navier-Stokes equations (e.g., Newton, Ossen, Stokes).

    Methods
    method()
        Select and initialize the FEM method (Stokes, Oseen, or Newton).
    run()
        Solve the Navier-Stokes system using the selected run strategy.
    apply_bc()
        Apply velocity and pressure boundary conditions.
    linear_system()
        Assemble the FEM system (bilinear and linear forms).
    solve()
        Solve the resulting linear system.
    lagrange_multiplier()
        Augment the linear system with a global pressure constraint.
    error()
        Compute errors in velocity and pressure.

    Notes
    This class is designed for stationary incompressible fluid flow problems using a mixed finite element method. 
    It supports velocity-pressure coupling and enforces pressure uniqueness through Lagrange multipliers. 

    Example
    >>> model = StationaryIncompressibleNSLFEMModel(equation, options)
    >>> model.__str__()
    """
    def __init__(self, pde, mesh=None, options=None):
        super().__init__(pbar_log=True, log_level="INFO")
        self.options = options
        self.pde = pde
        self.equation = StationaryStokes(pde)
        
        if mesh is None:
            if hasattr(pde, 'init_mesh'):
                self.mesh = pde.init_mesh(nx=options.get('nx', 8), ny=options.get('ny', 8))
            else:
                raise ValueError("Not found mesh!")
        else:
            self.mesh = mesh
        
        self.fem = self.method()
        

        if options is not None:
            self.solve.set(options['solve'])
            self.fem = self.method[options['method']]()
            self.run.set(options['run'])
            self.maxit = options.get('maxit', 5)
            self.maxstep = options.get('maxstep', 10)
            self.tol = options.get('tol', 1e-10)
            
    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the computational model configuration."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  equation       : {self.equation.__class__.__name__}\n"
        s += f"  pde            : {self.pde.__class__.__name__}\n"
        s += f"  method         : {self.method_str}\n"
        s += f"  run            : {self.run_str}\n"
        s += f"  solve          : {self.solve_str}\n"
        s += f"  maxsteps       : {self.maxstep}\n"
        s += f"  tol            : {self.tol}\n"
        if self.options.get("run") == "uniform_refine":
            s += f"  Max Refinement : {self.maxit}\n"
        s += ")"
        self.logger.info(f"\n{s}")
        
    @variantmethod("Stokes")
    def method(self): 
        """
        Use Newton iteration method to solve the Navier-Stokes equations.
        """
        from .simulation.fem.stationary_stokes.stokes import Stokes
        self.fem = Stokes(self.equation, self.mesh)
        self.method_str = "Stokes"
        return self.fem
    
    def update_mesh(self, mesh):
        """
        Update the mesh used in the model.
        """
        self.mesh = mesh
        self.fem.update_mesh(mesh)
    
    def update(self):   
        self.fem.update()
    
    def linear_system(self):
        """
        Assemble the linear system for the stationary navier-stokes finite element model.
        """
        BForm = self.fem.BForm()
        LForm = self.fem.LForm()
        return BForm, LForm
    
    
    
    @variantmethod('one_step')
    def run(self):
        self.run_str = 'one_step'
        BForm, LForm = self.linear_system()  
        self.fem.update()
        A = BForm.assembly() 
        # import ipdb
        # ipdb.set_trace()
        b = LForm.assembly()
        A, b = self.fem.apply_bc(A, b, self.pde)
        A, b = self.fem.lagrange_multiplier(A, b)
        x = self.solve(A, b)

        ugdof = self.fem.uspace.number_of_global_dofs()
        u = self.fem.uspace.function()
        p = self.fem.pspace.function()
        u[:] = x[:ugdof]
        p[:] = x[ugdof:-1] 
        return u, p

    @run.register('uniform_refine')
    def run(self, maxit = 5):
        self.run_str = 'uniform_refine'
        maxit = self.maxit if self.options is not None else maxit
        u_errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
        p_errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)
        for i in range(maxit):
            self.logger.info(f"number of cells: {self.mesh.number_of_cells()}")
            uh, ph = self.run['one_step']()
            uerror, perror = self.error(uh, ph)
            u_errorMatrix[0, i] = uerror
            p_errorMatrix[0, i] = perror
            order_u = bm.log2(u_errorMatrix[0,:-1]/u_errorMatrix[0,1:])
            order_p = bm.log2(p_errorMatrix[0,:-1]/p_errorMatrix[0,1:])
            self.pde.mesh.uniform_refine()
        self.logger.info(f"速度最终误差:" + ",".join(f"{uerror:.15e}" for uerror in u_errorMatrix[0,]))
        self.logger.info(f"order_u: " + ", ".join(f"{order_u:.15e}" for order_u in order_u))
        self.logger.info(f"压力最终误差:" + ",".join(f"{perror:.15e}" for perror in p_errorMatrix[0,]))  
        self.logger.info(f"order_p: " + ", ".join(f"{order_p:.15e}" for order_p in order_p))
        return uh, ph

    @variantmethod('direct')
    def solve(self, A, F, solver='scipy'):
        from fealpy.solver import spsolve
        self.solve_str = 'direct'
        return spsolve(A, F, solver = solver)

        
    def error(self, uh, ph):
        """
        Post-process the numerical solution to compute the error in L2 norm.
        """
        self.error_str = 'error'
        uerror = self.mesh.error(self.pde.velocity, uh)
        perror = self.mesh.error(self.pde.pressure, ph)
        return uerror, perror

    
