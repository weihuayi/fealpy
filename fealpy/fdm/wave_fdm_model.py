import matplotlib.pyplot as plt

from ..backend import backend_manager as bm
from ..mesh import UniformMesh
from ..fdm import LaplaceOperator, DirichletBC
from ..sparse import spdiags
from ..solver import spsolve
from ..model import PDEDataManager

class WaveFDMModel:
    """Finite Difference solver for second-order wave equations using theta-scheme
    
    Solves the second-order wave equation:
        d^2 u / dt^2 + a^2 A u = 0
    using a three-level theta-scheme in time. Supports mesh refinement studies
    and provides visualization capabilities for 1D and 2D problems.
    
    Parameters
    ----------
        example : str, optional, default='sincos'
            Name of the example problem to solve (must be available in PDEDataManager)
        maxit : int, optional, default=4
            Number of mesh refinement levels to perform
        ns : int, optional, default=20
            Initial number of segments per dimension
        solver : callable, optional, default=spsolve
            Linear system solver function (must have same interface as spsolve)
        theta : float, optional, default=0.5
            Weight parameter in [0,1] controlling implicitness:
            - 0: explicit scheme
            - 0.5: symmetric scheme (Crank-Nicolson type)
            - 1: fully implicit scheme
        nt : int, optional, default=400
            Number of time steps to use in the simulation
        
    Attributes
    ----------
        pde : PDEDataManager
            Manages PDE problem data including domain, initial/boundary conditions
        a : float
            Wave speed constant from the PDE
        mesh : UniformMesh
            Current computational mesh
        t0, t1 : float
            Start and end times of the simulation
        final_solutions : list
            Stores all time-step solutions for the final refinement level
        final_errors : ndarray
            Error metrics for each time step in the final refinement level
        u_nm1, u_nm2 : ndarray
            Solution vectors at previous time steps (n-1 and n-2)
        A, I : sparse matrix
            Discrete Laplace operator and identity matrix
    
    Methods
    -------
        run()
            Execute the full simulation with mesh refinement study
        show_solution(interval=100)
            Visualize the numerical solution animation
        show_error()
            Plot error evolution over time for the final refinement level
    """

    def __init__(self, example: str = 'sincos', maxit: int = 4, 
                 ns: int = 20, solver=spsolve, theta: float=0.5, nt: int = 400):
        """Initialize wave equation solver with problem parameters
        
        Parameters
        ----------
            example : str, optional, default='sincos'
                Name of pre-defined example problem
            maxit : int, optional, default=4
                Maximum number of mesh refinement iterations
            ns : int, optional, default=20
                Initial number of mesh segments per dimension
            solver : callable, optional, default=spsolve
                Linear system solver function
            theta : float, optional, default=0.5
                Time discretization parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit)
            nt : int, optional, default=400
                Number of time steps
        """
        self.pde = PDEDataManager('wave').get_example(example) 
        self.a = self.pde.speed()
        self.theta = theta
        self.nt = nt
        self.ns = ns
        self.maxit = maxit
        self.mesh = None
        self.t0, self.t1 = self.pde.duration()
        self.final_solutions = None
        self.final_errors = None

    def _generate_mesh(self, n):
        """Generate initial uniform mesh for the problem domain
        
            Parameters
            ----------
            n : int
                Number of segments per dimension in the initial mesh
        """
        domain = self.pde.domain()
        extent = [0, n] * self.pde.geo_dimension()
        self.mesh = UniformMesh(domain, extent)
    
    def _linear_system(self):
        """Assemble discrete Laplace operator and identity matrix
        
        Constructs the finite difference discretization of the Laplace operator
        and corresponding identity matrix for the current mesh.
        Results stored in self.A and self.I.
        """
        mesh = self.mesh
        A = LaplaceOperator(mesh).assembly()
        I = spdiags(bm.ones(A.shape[0], dtype=mesh.ftype), 0, *A.shape)
        self.A = A
        self.I = I

    def init_solutions(self):
        """Initialize solution vectors u^0 and u^1 using initial conditions
        
        Computes:
            u1 = (I - a^2 tau^2/2 A) u0 + tau * phi1
        where:
            - u0 is the initial condition u(x,0)
            - phi1 is the initial velocity du/dt(x,0)
            - tau is the time step size
        
        Applies Dirichlet boundary conditions at both initial time steps.
        """
        mesh = self.mesh
        # nodal values
        u0 = mesh.interpolate(self.pde.init_solution, etype='node').reshape(-1)
        # initial velocity phi1(x)
        phi1 = mesh.interpolate(self.pde.init_solution_t, etype='node').reshape(-1)
        # compute u1
        tau = (self.t1 - self.t0) / self.nt
        A = self.A
        I = self.I
        u1 = (I - 0.5 * tau**2 * self.a**2 * A) @ u0 + tau * phi1
        # apply Dirichlet BC at t0 and t1
        bd = mesh.boundary_node_flag()
        nodes = mesh.entity('node')
        u0[bd] = self.pde.dirichlet(nodes[bd], self.t0)
        u1[bd] = self.pde.dirichlet(nodes[bd], self.t0 + tau)
        self.u_nm2 = u0.copy()  # u^{n-2}
        self.u_nm1 = u1.copy()  # u^{n-1}

    def step(self, n, tau):
        """Advance solution one time step using theta-scheme
        
        Parameters
        ----------
            n : int
                Current time step index
            tau : float
                Time step size
            
        Solves the system:
            A0 u^n = A1 u^{n-1} + A2 u^{n-2}
        where coefficient matrices depend on theta scheme parameter.
        """
        A = self.A
        I = self.I
        theta = self.theta
        a2tau2 = self.a**2 * tau**2
        # define coefficient matrices
        A0 = I + theta * a2tau2 * A
        A1 = 2*I - (1 - 2*theta) * a2tau2 * A
        A2 = -(I + theta * a2tau2 * A)
        # right-hand side
        rhs = A1 @ self.u_nm1 + A2 @ self.u_nm2  # careful: use u^{n-1} & u^{n-2}
        # apply Dirichlet BC
        mesh = self.mesh
        t_n = self.t0 + n * tau
        A0, rhs = DirichletBC(mesh, lambda p: self.pde.dirichlet(p, t_n)).apply(A0, rhs)
        # solve linear system
        u_n = spsolve(A0, rhs, solver='scipy')
        # update history
        self.u_nm2, self.u_nm1 = self.u_nm1, u_n.copy()

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
        self._generate_mesh(self.ns)
        
        em = bm.zeros((1, self.nt+1))
        error = bm.zeros((3, self.maxit))

        for level in range(self.maxit):
            self._linear_system()
            self.init_solutions()
            sols = [self.u_nm2.copy()]
            sols.append(self.u_nm1.copy())

            if level == self.maxit:
                em[0, 0] = self.mesh.error(lambda p: self.pde.solution(p, t0),self.u_nm2,errortype='max')
                em[0, 1] = self.mesh.error(lambda p: self.pde.solution(p, t0+self.tau),self.u_nm1,errortype='max')
           
            for n in range(2, self.nt+1):
                t_n = t0 + n * self.tau
                self.step(n, self.tau)
                sols.append(self.u_nm1.copy())
                
                if level == self.maxit-1:
                    em[0, n] = self.mesh.error(lambda p: self.pde.solution(p, t_n), self.u_nm1, errortype='max')
            
            error[0, level], error[1, level], error[2, level] = self.mesh.error(
                lambda p: self.pde.solution(p, t_n), self.u_nm1, errortype='all')

            self.final_errors = em

            if level == self.maxit-1:            
                self.final_solutions = sols

            if level < self.maxit - 1:
                self.mesh.uniform_refine()

        self.final_errors = em
        print("最后时间步的误差(max,L2, H1/l1) ", error, "收敛阶: ", error[:, 0:-1] / error[:, 1:], sep='\n')

    def show_solution(self, interval=100):
        """Visualize numerical solution animation
        
        Displays an animated plot of the solution over time. Supports:
            - 1D: Line plot animation
            - 2D: Surface plot animation
        
        Parameters
        ----------
            interval : int, optional, default=100
                Delay between frames in milliseconds
            
        Note
        ----
            For dimensions > 2, prints a warning message as visualization
            is not supported.
        """
        GD = self.pde.geo_dimension()
        
        if GD > 2:
            print("Warning: Only 1D and 2D function visualization is supported. Current problem dimension is ", GD)
            return
        
        from matplotlib import animation
        mesh = self.mesh
        sol = self.final_solutions
        coords = mesh.entity('node')
        tau = (self.t1 - self.t0) / self.nt
        fig = plt.figure()

        if GD == 1:
            ax = fig.add_subplot(111)
            def update_1d(n):
                ax.clear()
                x = coords[:,0]
                ax.plot(x, sol[n])
                ax.set_ylim(sol[0].min(), sol[0].max())
                ax.set_title(f"t = {self.t0 + n*tau:.3f}")
                ax.set_xlabel('x')
                ax.set_ylabel('u')
            ani = animation.FuncAnimation(fig, update_1d, frames=len(sol), interval=interval)
        elif GD == 2:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            ax = fig.add_subplot(111, projection='3d')
            tri = mesh.entity('cell')
            x, y = coords[:,0], coords[:,1]
            def update_2d(n):
                ax.clear()
                ax.plot_trisurf(x, y, sol[n], triangles=tri, cmap='viridis')
                ax.set_title(f"t = {self.t0 + n*tau:.3f}")
            ani = animation.FuncAnimation(fig, update_2d, frames=len(sol), interval=interval)

        plt.show()

    def show_error(self):
        """Plot error evolution over time for final refinement level
        
        Creates a plot showing the maximum norm error versus time
        for the finest mesh resolution.
        """
        em = self.final_errors  # shape (1, nt+1)
        # time array
        nt = em.shape[1] - 1
        t0, t1 = self.pde.duration()
        times = [t0 + i*(t1 - t0)/nt for i in range(nt+1)]
        # select error sequence
        errs = em[0, :]
        labels = ['Max norm']
        plt.figure()
        plt.plot(times, errs, '-o', label=labels[0])
        plt.title(f'Error(Max norm) of the final mesh refinement')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        plt.show()
