from typing import Union
from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod, cartesian
from fealpy.model import ComputationalModel
from fealpy.mesh import Mesh
from fealpy.utils import timer
from fealpy.cfd.equation import StationaryIncompressibleNS
from scipy.interpolate import griddata

class StationaryIncompressibleNSLFEMModel(ComputationalModel):
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
        self.equation = StationaryIncompressibleNS(pde)
        
        if mesh is None:
            if hasattr(pde, 'init_mesh'):
                self.mesh = pde.init_mesh()
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
        
    @variantmethod("Newton")
    def method(self): 
        """
        Use Newton iteration method to solve the Navier-Stokes equations.
        """
        from .simulation.fem.stationary_incompressible_ns import Newton
        self.fem = Newton(self.equation, self.mesh)
        self.method_str = "Newton"
        return self.fem
    
    def update_mesh(self, mesh):
        """
        Update the mesh used in the model.
        """
        self.mesh = mesh
        self.fem.update_mesh(mesh)

    @method.register("Ossen")
    def method(self): 
        """
        Use Oseen iteration method to solve the Navier-Stokes equations.
        """
        from .simulation.fem.stationary_incompressible_ns import Ossen
        self.fem = Ossen(self.equation, self.mesh)
        self.method_str = "Ossen"
        return self.fem
    
    @method.register("Stokes")
    def method(self): 
        """
        Use Stokes iteration method to solve the system.
        """
        from .simulation.fem.stationary_incompressible_ns import Stokes
        self.fem = Stokes(self.equation, self.mesh)
        self.method_str = "Stokes"
        return self.fem
    
    def update(self, u0):   
        self.fem.update(u0)
    
    def linear_system(self):
        """
        Assemble the linear system for the stationary navier-stokes finite element model.
        """
        BForm = self.fem.BForm()
        LForm = self.fem.LForm()
        return BForm, LForm
    
    
    @variantmethod('main')
    def run(self, maxstep=1000, tol=1e-10):
        self.run_str = 'main'
        maxstep = self.maxstep if self.options is not None else maxstep
        tol = self.tol if self.options is not None else tol
        uh0 = self.fem.uspace.function()
        ph0 = self.fem.pspace.function()
        
        for i in range(maxstep):
            uh1, ph1 = self.run['one_step'](uh0)
            res_u = self.mesh.error(uh0, uh1)
            res_p = self.mesh.error(ph0, ph1)
            self.logger.info(f"res_u: {res_u}, res_p: {res_p}")
            if res_u + res_p < tol:
                self.logger.info(f"Converged at iteration {i+1}")
                break 
            uh0[:] = uh1
            ph0[:] = ph1
        uerror, perror = self.error(uh1, ph1) 
        self.logger.info(f"Final error: uerror = {uerror}, perror = {perror}")
        return uh1, ph1
    
    @run.register('one_step')
    def run(self, uh):
        self.run_str = 'one_step'
        BForm, LForm = self.linear_system()  
        self.fem.update(uh)
        A = BForm.assembly() 
        b = LForm.assembly()
        A, b = self.fem.apply_bc(A, b, self.pde)
        if self.equation.pressure_neumann == True:
            A, b = self.fem.lagrange_multiplier(A, b)
        x = self.solve(A, b)

        ugdof = self.fem.uspace.number_of_global_dofs()
        u = self.fem.uspace.function()
        p = self.fem.pspace.function()
        u[:] = x[:ugdof]
        if self.equation.pressure_neumann == True:
            p[:] = x[ugdof:-1]
        else:
            p[:] = x[ugdof:]
        return u, p

    @run.register('uniform_refine')
    def run(self, maxit = 5, maxstep = 1000, tol = 1e-10):
        self.run_str = 'uniform_refine'
        maxit = self.maxit if self.options is not None else maxit
        maxstep = self.maxstep if self.options is not None else maxstep
        tol = self.tol if self.options is not None else tol
        for i in range(maxit):
            self.logger.info(f"number of cells: {self.mesh.number_of_cells()}")
            uh1, ph1 = self.run['main'](maxstep, tol)
            self.mesh.uniform_refine()
        return uh1, ph1

    @variantmethod('direct')
    def solve(self, A, F, solver='scipy'):
        from fealpy.solver import spsolve
        self.solve_str = 'direct'
        return spsolve(A, F, solver = solver)

    @variantmethod('L2')    
    def error(self, uh, ph):
        """
        Post-process the numerical solution to compute the error in L2 norm.
        """
        self.error_str = 'error'
        uerror = self.mesh.error(self.pde.velocity, uh)
        perror = self.mesh.error(self.pde.pressure, ph)
        return uerror, perror
    
    @error.register('benchmark')
    def error(self, uh, ph):
        """
        Compute the error in benchmark form.
        """
        self.error_str = 'benchmark'
        fem = self.fem
        mesh = self.mesh
        location = mesh.location
        ipoints = fem.uspace.interpolation_points()
        qf = mesh.quadrature_formula(q=4, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        vd = fem.uspace.function()
        vl = fem.uspace.function()
        vd[:len(ipoints)][self.pde.is_obstacle_boundary(ipoints)] = 1.0
        vl[len(ipoints):][self.pde.is_obstacle_boundary(ipoints)] = 1.0
        
        cellmeasure = self.mesh.entity_measure("cell")
        p = ph(bcs = bcs)
        grad_vd = self.fem.uspace.grad_value(uh = vd, 
                                             bc = bcs)
        grad_uh = self.fem.uspace.grad_value(uh = uh, 
                                             bc = bcs)
        cd = self.pde.mu * bm.einsum('n, knij, knij, k-> ', ws, grad_uh, grad_vd, cellmeasure) 
        cd += self.pde.rho * bm.einsum('n, knj, knij, kni, k -> ',ws, uh(bcs = bcs), 
                                                    grad_uh,
                                                    vd(bcs = bcs), cellmeasure)  
        cd -= bm.einsum('n, kn, knii, k -> ', ws, p, grad_vd, cellmeasure) 

        grad_vl = self.fem.uspace.grad_value(uh = vl, 
                                             bc = bcs)
        grad_uh = self.fem.uspace.grad_value(uh = uh, 
                                             bc = bcs)
        cl = self.pde.mu * bm.einsum('n, knij, knij, k-> ', ws, grad_uh, grad_vl, cellmeasure)
        cl += self.pde.rho * bm.einsum('n, knj, knij, kni, k -> ', ws, 
                                                                    uh(bcs = bcs), 
                                                                    grad_uh, 
                                                                    vl(bcs = bcs),
                                                                    cellmeasure)
        cl -= bm.einsum('n, knii, kn, k -> ', ws, grad_vl, p, cellmeasure)
        
        point0 = bm.array([[0.15, 0.2]])
        point1 = bm.array([[0.25, 0.2]])
        index0 = location(points=point0)
        index1 = location(points=point1)

        def get_bcs(point, index):
            node_points = mesh.entity("node")
            c2n = mesh.cell_to_node()
            cell = node_points[c2n][index][0]
            S = 0.5 * bm.cross(cell[1]-cell[0], cell[2]-cell[0])
            lambda1 = 0.5 * bm.cross(cell[1]-point[0], cell[2]-point[0]) / S
            lambda2 = 0.5 * bm.cross(cell[2]-point[0], cell[0]-point[0]) / S
            lambda3 = 1.0 - lambda1 - lambda2
            bcs = bm.array([[lambda1, lambda2, lambda3]])
            return bcs
        
        bcs0 = get_bcs(point=point0, index = index0)
        bcs1 = get_bcs(point=point1, index = index1)

        cd = -500 * cd
        cl = -500 * cl
        delta_p = ph(bcs = bcs0, index = index0) - ph(bcs = bcs1, index = index1)
        return cd, cl, delta_p
    
