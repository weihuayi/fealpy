import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.mesh import UniformMesh 
from fealpy.fdm import LaplaceOperator   
from fealpy.fdm import DirichletBC
from fealpy.sparse import spdiags
from fealpy.solver import spsolve

class WaveFDMModel:
    """
    Finite Difference solver for wave equations on uniform grids.
    This class implements both explicit and implicit time integration schemes,
    supports uniform mesh refinement, error analysis, and solution visualization.
    
    Parameters
    ----------
    pde : PDEData object
        Provides domain, initial conditions, source term, Dirichlet BC, exact solution, etc.
    scheme : str, optional
        Time-stepping scheme: 'explicit' or 'implicit' (default 'explicit').
    """
    def __init__(self, pde, nt=400, maxit=4, ns=10, scheme='explicit'):
        """
        Initialize the WaveFDMModel.
        
        Parameters
        ----------
        pde : PDEData object
            Provides problem data (initial conditions, source, dirichlet, solution).
        nt : int, optional
            Number of time steps (default 400).
        maxit : int, optional
            Number of uniform mesh refinements (default 4).
        ns : int or list of int, optional
            Initial number of segments per dimension (default 10).
        scheme : {'explicit', 'implicit'}, optional
            Time-stepping scheme (default 'explicit').
        """
        self.pde = pde
        self.scheme = scheme.lower()
        self.nt = nt
        self.maxit = maxit
        self.ns = ns
        self.mesh = None
        self.t0, self.t1 = pde.duration()
        self.all_solutions = {}
        self.all_errors = {}
        
    def _generate_mesh(self, n):
        """Generate initial uniform mesh
        
        Parameters
            n : int
                Number of segments per dimension
        """
        domain = self.pde.domain()
        extent = [0, n] * self.pde.geo_dimension()
        self.mesh = UniformMesh(domain, extent)
        
    def assemble_matrices(self):
        mesh = self.mesh
        self.A = LaplaceOperator(mesh).assembly()
        self.I = spdiags(bm.ones(self.A.shape[0], dtype=mesh.ftype), 0, 
                        self.A.shape[0], self.A.shape[1])
        
    def init_solution(self):
        mesh = self.mesh
        # Initial displacement
        self.u0 = mesh.interpolate(self.pde.init_displacement, etype='node').reshape(-1)
        # Initial velocity
        self.v0 = mesh.interpolate(self.pde.init_velocity, etype='node').reshape(-1)
        # First time step for explicit scheme
        if self.scheme == 'explicit':
            self.u1 = self.u0 + self.tau * self.v0 + 0.5 * self.tau**2 * (
                self.A @ self.u0 + mesh.interpolate(
                    lambda p: self.pde.source(p, self.t0), etype='node').reshape(-1))
        
    def explicit_step(self, n, tau):
        mesh = self.mesh
        t = self.t0 + n * tau
        F = mesh.interpolate(lambda p: self.pde.source(p, t), etype='node').reshape(-1)
        
        # Central difference scheme
        u_new = 2 * self.u1 - self.u0 + tau**2 * (self.A @ self.u1 + F)
        
        # Apply boundary conditions
        bd = mesh.boundary_node_flag()
        nodes = mesh.entity('node')
        u_new[bd] = self.pde.dirichlet(nodes[bd], t)
        
        # Update solutions
        self.u0, self.u1 = self.u1, u_new
        
    def implicit_step(self, n, tau):
        mesh = self.mesh
        t = self.t0 + n * tau
        F = mesh.interpolate(lambda p: self.pde.source(p, t), etype='node').reshape(-1)
        
        # Implicit scheme matrix
        S = self.I - tau**2 * self.A
        b = 2 * self.u1 - self.u0 + tau**2 * F
        
        # Apply boundary conditions
        S, b = DirichletBC(mesh, lambda p: self.pde.dirichlet(p, t)).apply(S, b)
        
        # Solve system
        u_new = spsolve(S, b, solver='scipy')
        
        # Update solutions
        self.u0, self.u1 = self.u1, u_new
        
    def step(self, n, tau):
        if self.scheme == 'explicit':
            self.explicit_step(n, tau)
        else:
            self.implicit_step(n, tau)
            
    def run(self):
        """
        Execute time-stepping on successively refined meshes.
        """
        t0, t1 = self.pde.duration()
        self.tau = (t1 - t0) / self.nt
        
        self.generate_mesh()
        prev_err_end = None
        
        for level in range(self.maxit):
            self.assemble_matrices()
            self.init_solution()
            
            sols = [self.u0.copy(), self.u1.copy()]  # Store initial two steps
            em = bm.zeros((3, self.nt+1), dtype=float)
            
            # Compute initial errors
            em[:, 0] = self.mesh.error(lambda p: self.pde.solution(p, t0), 
                                     self.u0, errortype='all')
            em[:, 1] = self.mesh.error(lambda p: self.pde.solution(p, t0+self.tau), 
                                     self.u1, errortype='all')
            
            for n in range(2, self.nt+1):
                t_n = t0 + n * self.tau
                self.step(n, self.tau)
                sols.append(self.u1.copy())
                em[:, n] = self.mesh.error(
                    lambda p: self.pde.solution(p, t_n),
                    self.u1,
                    errortype='all'
                )
                
            self.all_solutions[level] = sols
            self.all_errors[level] = em
            
            err_end = em[0, -1]  # max-norm at final time
            if prev_err_end is not None:
                ratio = prev_err_end / err_end
                print(f"refinement {level}: final error={err_end:.2e}, ratio={ratio:.2f}")
            else:
                print(f"refinement {level}: final error={err_end:.2e}")
                
            prev_err_end = err_end
            if level < self.maxit - 1:
                self.mesh.uniform_refine()
                
    def show_solution(self, zlim=None, interval=100):
        """
        Animate the 2D surface of u(x,y) over time.
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        
        mesh = self.mesh
        sol = self.all_solutions[self.maxit - 1]
        dt = self.tau
        
        node = mesh.entity('node')
        x, y = node[:, 0], node[:, 1]
        tri = mesh.entity('cell')
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if zlim:
            ax.set_zlim(*zlim)
            
        surf = ax.plot_trisurf(x, y, sol[0], triangles=tri, cmap='viridis')
        
        def update(frame):
            ax.clear()
            t = self.t0 + frame * dt
            ax.set_title(f"Time = {t:.2f}")
            ax.plot_trisurf(x, y, sol[frame], triangles=tri, cmap='viridis')
            if zlim:
                ax.set_zlim(*zlim)
            return ax.collections
            
        ani = animation.FuncAnimation(
            fig, update, frames=len(sol), interval=interval, blit=False
        )
        plt.show()
        
    def show_error(self, level=0, error_type=0):
        """
        Plot a single error metric over time.
        """
        import matplotlib.pyplot as plt
        
        em = self.all_errors[level]
        nt = em.shape[1] - 1
        t0, t1 = self.pde.duration()
        times = [t0 + i*(t1 - t0)/nt for i in range(nt+1)]
        errs = em[error_type, :]
        
        labels = ['Max norm', 'L2 norm', 'H1 norm']
        plt.figure()
        plt.plot(times, errs, '-o', label=labels[error_type])
        plt.title(f'{labels[error_type]} vs Time (level={level})')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        plt.show()
