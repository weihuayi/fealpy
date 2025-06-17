import matplotlib.pyplot as plt

from ..backend import backend_manager as bm
bm.set_backend('numpy')

from ..sparse import spdiags
from ..solver import spsolve

from ..model import PDEDataManager
from ..mesh import UniformMesh     
from . import DirichletBC, ConvectionOperator



class HyperbolicFDMModel:
    """Finite Difference solver for parabolic PDEs on uniform grids.
    
    This class implements time integration using Forward Euler, Backward Euler
    or Crank–Nicolson schemes, supports uniform mesh refinement, error
    analysis across time and space refinements, as well as solution and
    error visualization.
    
    Parameters
    ----------
        pde : PDEData object
            Provides problem data (initial conditions, source, dirichlet, solution).
        tau : float
            Time step size for marching.
        maxit : int, optional, default=4
            Number of uniform mesh refinements.
        ns : int or list of int, optional, default=20
            Initial number of segments per dimension.
        nt : int, optional, default=400
            Number of time steps.
        scheme : {'forward','backward','cn'}, optional, default='backward'
            Time-stepping scheme: forward Euler, backward Euler,
            or Crank–Nicolson.
        method : optional'upwind_const_1' or 'central_const_2',
            the meaning is the assembly method for the convection term.

    
    Attributes
    ----------
        pde : PDEData object
            Provides domain, initial conditions, source term, Dirichlet BC, exact solution, etc.
        maxit : int
            Number of uniform mesh refinements.
        ns : int or list of int
            Initial number of segments per dimension.
        solver : function
            Solver function for linear systems.
        scheme : str
            Time-stepping scheme.
        method : the meaning is the assembly method for the convection term.
        nt : int
            Number of time steps.
        mesh : UniformMesh object
            The computational mesh.
        t0, t1 : float
            Start and end times of the simulation.
        final_solutions : ndarray
            Array of solutions at the final time step for each refinement level.
        final_errors : ndarray
            Array of errors at the final time step for each refinement level.
    """
    
    def __init__(self, example: str = 'sinsin', maxit: int = 4, 
                 ns: int = 20, solver=spsolve, nt: int = 400, 
                 scheme: str='backward', method: str ='upwind_const_1'):
        """
        Initialize the ParabolicFDMModel.
        
        Parameters
        ----------
            pde : PDEData object
                Provides problem data (initial conditions, source, dirichlet, solution).
            tau : float
                Time step size for marching.
            maxit : int, optional, default=4
                Number of uniform mesh refinements.
            ns : int or list of int, optional, default=20
                Initial number of segments per dimension.
            nt : int, optional, default=400
                Number of time steps.
            scheme : {'forward','backward','cn'}, optional, default='backward'
                Time-stepping scheme: forward Euler, backward Euler,
                or Crank–Nicolson.
        """
        self.pde = PDEDataManager('hyperbolic').get_example(example) 
        self.maxit = maxit
        self.ns = ns
        self.solver = spsolve
        self.scheme = scheme.lower()
        self.nt = nt
        self.maxit = maxit
        self.method = method
        self.mesh = None
        self.t0, self.t1 = self.pde.duration()
        self.final_solutions = None
        self.final_errors    = None

        
    def _generate_mesh(self, n):
        """Generate initial uniform mesh.
        
        Parameters
        ----------
            n : int
                Number of segments per dimension.
        """
        domain = self.pde.domain()
        extent = [0, n] * self.pde.geo_dimension()
        self.mesh = UniformMesh(domain, extent)
    
    def _linear_system(self):
        mesh = self.mesh
        pde = self.pde

        if hasattr(pde, 'convection_coef'):
            if self.method == 'upwind_const_1':
                A = ConvectionOperator(mesh=mesh, convection_coef=pde.convection_coef, 
                                    method=self.method).assembly()
            elif self.method == 'central_const_2':
                co = ConvectionOperator(mesh=mesh, convection_coef=pde.convection_coef, 
                                    method=self.method)
                A = co.assembly_central_const()
            else:
                raise(f"There is no method {self.method}, \
                      only methods 'upwind_const_1' and 'central_const_2'.")
        I = spdiags(bm.ones(A.shape[0], dtype=mesh.ftype), 0, A.shape[0], A.shape[1])
        self.A = A
        self.I = I


    def init_solution(self):
        """Initialize the solution at the initial time step."""
        uh0 = self.mesh.interpolate(self.pde.init_solution, etype='node').reshape(-1)
        self.uh = uh0.copy()


    def step(self, n, tau):
        """Perform a single time step.
        
        Parameters
        ----------
            n : int
                Current time step index.
            tau : float
                Time step size.
        """
        mesh = self.mesh
        A    = self.A
        I    = self.I
        uh   = self.uh
        t    = self.t0 + n*tau
        F = mesh.interpolate(lambda p: self.pde.source(p, t), etype='node').reshape(-1)

        if self.scheme == 'forward':
            uh_new = uh - tau * (A@uh) + tau * F
            bd = mesh.boundary_node_flag()
            nodes = mesh.entity('node')
            uh_new[bd] = self.pde.dirichlet(nodes[bd], t)
            self.uh = uh_new

        elif self.scheme == 'backward':
            S = I + tau*A
            b = uh + tau*F
            bd = mesh.boundary_node_flag()
            S, b = DirichletBC(mesh, lambda p: self.pde.dirichlet(p, t)).apply(S, b)
            self.uh = spsolve(S, b, solver='scipy')

        elif self.scheme == 'cn':
            S = I + 0.5*tau*A
            b = (I - 0.5*tau*A)@uh + tau*F
            bd = mesh.boundary_node_flag()
            S, b = DirichletBC(mesh, lambda p: self.pde.dirichlet(p, t)).apply(S, b)
            self.uh = spsolve(S, b, solver='scipy')

        else:
            raise ValueError(f"Unknown scheme {self.scheme}")
        
    def run(self):
        """Execute time-stepping on successively refined meshes.
            
            1. Generate initial mesh with n0 segments/direction.
            2. For each refinement level:
                - Assemble matrices, init solution
                - March forward for nt steps
                - Record solution/time and errors
                - Uniformly refine mesh for next level
            3. Print final-time errors and error ratios.
        """
        t0, t1 = self.t0, self.t1
        self.tau = (t1 - t0) / self.nt
        self._generate_mesh(n=self.ns)

        em = bm.zeros((1, self.nt+1))
        error = bm.zeros((3, self.maxit))

        for level in range(self.maxit):
            self._linear_system()
            self.init_solution()
            sols = [self.uh.copy()]

            if level == self.maxit:
                em[0, 0] = self.mesh.error(lambda p: self.pde.solution(p, t0), self.uh, errortype='max')
                
            for n in range(1,self.nt+1):
                t_n = t0 + n * self.tau
                self.step(n, self.tau)
                sols.append(self.uh.copy())

                if level == self.maxit-1:
                    em[0, n] = self.mesh.error(lambda p: self.pde.solution(p, t_n), self.uh, errortype='max')
            
            error[0, level], error[1, level], error[2, level] = self.mesh.error(
                lambda p: self.pde.solution(p, t_n), self.uh, errortype='all')

            if level == self.maxit-1:            
                self.final_solutions =  sols
 

            if level < self.maxit - 1:
                self.mesh.uniform_refine()

        self.final_errors = em
        print("最后时间步的误差(max,L2, H1/l1): ", error, "收敛阶: ", error[:, 0:-1] / error[:, 1:], sep='\n')

    def show_solution(self, zlim=None, interval=100):
        """Display numerical solution visualization.
        
        Note
        ----
            Only supports 1D and 2D problems. For 1D: plots solution curve.
            For 2D: shows 3D surface plot. Higher dimensions will print warning.
        """
        
        GD = self.pde.geo_dimension()
        
        if GD > 2:
            print("Warning: Only 1D and 2D function visualization is supported. Current problem dimension is ", GD)
            return
        
        from matplotlib import animation

        mesh = self.mesh
        sol = self.final_solutions  # Ensure sol is [n_time_steps, n_nodes] array
        n_sol = len(sol)
        node = mesh.entity('node')
        dim = node.shape[1]  # Mesh dimension (1, 2, or 3)

        # Time step size
        dt = self.tau if self.tau is not None else (self.pde.duration()[1] - self.t0) / n_sol

        # Create figure
        fig = plt.figure(figsize=(10, 10))
        
        # ===== 1D Case =====
        if GD == 1:
            ax = fig.add_subplot(111)
            x = node[:, 0]
            line, = ax.plot(x, sol[0], 'b-', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.set_title(f"Time = {self.t0:.2f}")

            def update(frame):
                t = self.t0 + frame * dt
                line.set_ydata(sol[frame])
                ax.set_title(f"Time = {t:.2f}")
                return line
            ani = animation.FuncAnimation(fig, update, frames=len(sol), interval=interval)


        # ===== 2D Case =====
        elif GD == 2:
            ax = fig.add_subplot(111, projection='3d')
            x, y = node[:, 0], node[:, 1]
            tri = mesh.entity('cell')  # Triangle mesh (only for 2D)
            surf = ax.plot_trisurf(x, y, sol[0], triangles=tri, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u(x,y)')
            
            if zlim:
                ax.set_zlim(*zlim)

            def update_2d(frame):
                ax.clear()
                t = self.t0 + frame * dt
                ax.plot_trisurf(x, y, sol[frame], triangles=tri, cmap='viridis')
                ax.set_title(f"Time = {t:.2f}")
                if zlim:
                    ax.set_zlim(*zlim)
                return ax.collections

        # Create animation
            ani = animation.FuncAnimation(fig, update_2d, frames=n_sol, interval=interval) 
        
        plt.show()            

    def show_error(self, error_type=0):
        """Plot a single error metric over time for a given refinement level.
        
        Parameters
        ----------
            error_type : int, optional, default=0
                Index of error to plot (0:max, 1:L2, 2:H1).
        """
        em = self.final_errors  # shape (1, nt+1)
        # Time array
        nt = em.shape[1] - 1
        t0, t1 = self.pde.duration()
        times = [t0 + i*(t1 - t0)/nt for i in range(nt+1)]
        # Select error sequence
        errs = em[error_type, :]
        labels = ['Max norm']
        plt.figure(figsize=(10, 10))
        plt.plot(times, errs, '-o', label=labels[error_type])
        plt.title(f'Error(Max norm) of the final mesh refinement')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        plt.show()
