
import matplotlib.pyplot as plt

from ..backend import backend_manager as bm
bm.set_backend('numpy')

from ..sparse import spdiags
from ..solver import spsolve

from ..model import PDEDataManager
from ..mesh import UniformMesh  
from . import LaplaceOperator, DirichletBC
from . import DiffusionOperator, ConvectionOperator, ReactionOperator



class ParabolicFDMModel:
    """Finite Difference solver for parabolic PDEs using various time schemes
    
    Solves parabolic PDEs of the form:
        du/dt + A u = f
        where A may contain diffusion, convection and reaction terms. Supports:
        - Forward/Backward Euler and Crank-Nicolson time schemes
        - Uniform mesh refinement studies
        - Error analysis in multiple norms(max, L2, H1\l2)
        - Solution visualization for 1D/2D problems
    
    Parameters
    ----------
        example : str, optional, default='sinsin'
            Name of pre-defined example problem
        maxit : int, optional, default=4
            Number of mesh refinement levels
        ns : int, optional, default=20  
            Initial number of segments per dimension
        solver : callable, optional, default=spsolve
            Linear system solver function
        nt : int, optional, default=400
            Number of time steps
        method : optional'upwind_const_1' or 'central_const_2',
            the meaning is the assembly method for the convection term.
        scheme : str, optional, default='backward'
            Time stepping scheme:
            - 'forward': Forward Euler (explicit)
            - 'backward': Backward Euler (implicit)
            - 'cn': Crank-Nicolson (implicit)
        
    Attributes
    ----------
        pde : PDEDataManager
            Manages PDE problem data (domain, ICs, BCs, exact solution)
        mesh : UniformMesh
            Current computational mesh
        method : the meaning is the assembly method for the convection term.
        t0, t1 : float
            Start and end times of simulation
        final_solutions : list
            Solution vectors at each time step for final refinement level
        final_errors : ndarray
            Error metrics over time for final refinement level
        A : sparse matrix
            Discretized differential operator
        I : sparse matrix  
            Identity matrix of same size as A
        uh : ndarray
            Current solution vector
    """

    def __init__(self, example: str = 'sinsin', maxit: int = 4, 
                 ns: int = 20, solver=spsolve, nt: int = 400, 
                 scheme: str='backward', method: str ='upwind_const_1'):
    
        self.pde = PDEDataManager('parabolic').get_example(example) 
        self.maxit = maxit
        self.ns = ns
        self.solver = spsolve
        self.scheme = scheme.lower()
        self.method = method
        self.nt = nt
        self.maxit = maxit
        self.mesh = None
        self.t0, self.t1 = self.pde.duration()
        self.final_solutions = None
        self.final_errors = None

    def _generate_mesh(self, n):
        """Generate uniform mesh for problem domain
        
        Parameters
        ----------
            n : int
                Number of segments per dimension
        """
        domain = self.pde.domain()
        extent = [0, n] * self.pde.geo_dimension()
        self.mesh = UniformMesh(domain, extent)
    
    def _linear_system(self):
        """Assemble discrete differential operator and identity matrix
        
        Constructs the finite difference discretization matrix A based on:
            - Diffusion term (Laplace by default)
            - Optional convection term
            - Optional reaction term
        
        Results stored in self.A and self.I.
        """
        mesh = self.mesh
        pde = self.pde

        if hasattr(pde, 'diffusion_coef'): 
            A = DiffusionOperator(mesh=mesh, diffusion_coef=pde.diffusion_coef).assembly()
        else:
            A = LaplaceOperator(mesh).assembly()
        
        if hasattr(pde, 'convection_coef'):
            if self.method == 'upwind_const_1':
                A += ConvectionOperator(mesh=mesh, convection_coef=pde.convection_coef, 
                                    method=self.method).assembly()
            elif self.method == 'central_const_2':
                co = ConvectionOperator(mesh=mesh, convection_coef=pde.convection_coef, 
                                    method=self.method)
                A += co.assembly_central_const()
            else:
                raise(f"There is no method {self.method}, \
                      only methods 'upwind_const_1' and 'central_const_2'.")
        
        if hasattr(pde, 'reaction_coef'):  
            A += ReactionOperator(mesh=mesh, reaction_coef=pde.reaction_coef).assembly()

        I = spdiags(bm.ones(A.shape[0], dtype=mesh.ftype), 0, A.shape[0], A.shape[1])
        self.A = A
        self.I = I

    def init_solution(self):
        """Initialize solution using initial condition
        
        Interpolates initial condition onto mesh nodes and stores in self.uh.
        """
        uh0 = self.mesh.interpolate(self.pde.init_solution, etype='node').reshape(-1)
        self.uh = uh0.copy()

    def step(self, n, tau):
        """Advance solution one time step using selected scheme
        
        Parameters
        ----------
            n : int
                Current time step index
            tau : float
                Time step size
                
        Notes
        -----
            Implements three schemes:
                - Forward Euler (explicit)
                - Backward Euler (implicit)
                - Crank-Nicolson (implicit)
            Applies Dirichlet BCs appropriately for each scheme.
        """
        mesh = self.mesh
        A = self.A
        I = self.I
        uh = self.uh
        t = self.t0 + n*tau
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
        """Execute full simulation with mesh refinement study
        
        Performs:
            1. Initial mesh generation
            2. For each refinement level:
                - Matrix assembly and solution initialization
                - Time stepping through all time steps
                - Error computation and storage
                - Mesh refinement for next level
            3. Outputs final error statistics and convergence rates
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
                self.final_solutions = sols

            if level < self.maxit - 1:
                self.mesh.uniform_refine()

        self.final_errors = em
        print("最后时间步的误差(max,L2, H1/l1): ", error, "收敛阶: ", error[:, 0:-1] / error[:, 1:], sep='\n')

    def show_solution(self, zlim=None, interval=100):
        """Visualize numerical solution animation
        
        Parameters
        ----------
            zlim : tuple, optional
                (min, max) limits for z-axis in 2D plots
            interval : int, optional, default=100
                Delay between frames in milliseconds
            
        Notes
        -----
            Supports:
            - 1D: animated line plot
            - 2D: animated surface plot
            For dimensions > 2, prints warning message.
        """
        GD = self.pde.geo_dimension()
        
        if GD > 2:
            print("Warning: Only 1D and 2D function visualization is supported. Current problem dimension is ", GD)
            return
        
        from matplotlib import animation

        mesh = self.mesh
        sol = self.final_solutions
        n_sol = len(sol)
        node = mesh.entity('node')


        dt = self.tau if self.tau is not None else (self.pde.duration()[1] - self.t0) / n_sol

        fig = plt.figure(figsize=(8, 8))
        
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
                return line,
            ani = animation.FuncAnimation(fig, update, frames=n_sol, interval=interval)

        elif GD == 2:
            ax = fig.add_subplot(111, projection='3d')
            x, y = node[:, 0], node[:, 1]
            tri = mesh.entity('cell')
            surf = ax.plot_trisurf(x, y, sol[0], triangles=tri, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u(x,y)')
            if zlim:
                ax.set_zlim(*zlim)

            def update(frame):
                ax.clear()
                t = self.t0 + frame * dt
                ax.plot_trisurf(x, y, sol[frame], triangles=tri, cmap='viridis')
                ax.set_title(f"Time = {t:.2f}")
                if zlim:
                    ax.set_zlim(*zlim)
                return ax.collections

            ani = animation.FuncAnimation(
                fig, update, frames=n_sol, interval=interval) 
        
        plt.show()            

    def show_error(self):
        """Plot error evolution over time for final refinement level
        
        Parameters
        ----------
        error_type : int, optional, default=0
            Error metric to plot:
            - 0: maximum norm
            - 1: L2 norm
            - 2: H1 norm
        """
        em = self.final_errors
        nt = em.shape[1] - 1
        t0, t1 = self.pde.duration()
        times = [t0 + i*(t1 - t0)/nt for i in range(nt+1)]
        errs = em.reshape(-1)
        labels = ['Max norm']
        
        plt.figure(figsize=(8, 8))
        plt.plot(times, errs, '-o', label=labels)
        plt.title(f'Error({labels}) of the final mesh refinement')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        plt.show()