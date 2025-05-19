
import matplotlib.pyplot as plt

from ..backend import backend_manager as bm
bm.set_backend('numpy')

from ..sparse import spdiags
from ..solver import spsolve

from ..model import PDEDataManager
from ..mesh import UniformMesh 
from ..fdm import LaplaceOperator   
from ..fdm import DirichletBC
from .diffusion_operator import DiffusionOperator
from .convection_operator import ConvectionOperator
from .reaction_operator import ReactionOperator


class HyperbolicFDMModel:
    """
    Finite Difference solver for parabolic PDEs on uniform grids.
    This class implements time integration using Forward Euler, Backward Euler
    or Crank–Nicolson schemes, supports uniform mesh refinement, error
    analysis across time and space refinements, as well as solution and
    error visualization.
    Parameters
    ----------
    pde : PDEData object
        Provides domain, initial conditions, source term, Dirichlet BC, exact solution, etc.
    scheme : str, optional
        Time-stepping scheme: 'forward', 'backward', or 'cn' (default 'backward').
    """
    
    def __init__(self, example: str = 'sinsin', maxit: int = 4, 
                 ns: int = 20, solver=spsolve, nt: int = 400,  scheme: str='backward'):
        """
        Initialize the ParabolicFDMModel.
        Parameters
        ----------
        pde : PDEData object
            Provides problem data (initial conditions, source, dirichlet, solution).
        tau : float
            Time step size for marching.
        maxit : int, optional
            Number of uniform mesh refinements (default 4).
        ns : int or list of int, optional
            Initial number of segments per dimension (default 10).
        nt : int, optional
            Number of time steps (default 400).
        scheme : {'forward','backward','cn'}, optional
            Time-stepping scheme: forward Euler, backward Euler,
            or Crank–Nicolson (default 'backward').
        """
        self.pde = PDEDataManager('hyperbolic').get_example(example) 
        self.maxit = maxit
        self.ns = ns
        self.solver = spsolve
        self.scheme = scheme.lower()
        self.nt = nt
        self.maxit = maxit
        self.mesh = None
        self.t0, self.t1 = self.pde.duration()
        self.all_solutions = {}
        self.all_errors    = {}

    
        
    def _generate_mesh(self, n):
        """Generate initial uniform mesh
        
        Parameters
        ----------
        n : int
            Number of segments per dimension
        """
        domain = self.pde.domain()
        extent = [0, n] * self.pde.geo_dimension()
        self.mesh = UniformMesh(domain, extent)
    
    def _linear_system(self):
        mesh = self.mesh
        pde = self.pde

        if hasattr(pde, 'diffusion_coef'): 
            A = DiffusionOperator(mesh=mesh, diffusion_coef=pde.diffusion_coef).assembly()
        else:
            A = LaplaceOperator(mesh).assembly()

        I = spdiags(bm.ones(A.shape[0], dtype=mesh.ftype), 0, A.shape[0], A.shape[1])
        self.A = A
        self.I = I

        if hasattr(pde, 'convection_coef'): 
            A += ConvectionOperator(mesh=mesh, convection_coef=pde.convection_coef).assembly()
        if hasattr(pde, 'reaction_coef'):  
            A += ReactionOperator(mesh=mesh, reaction_coef=pde.reaction_coef).assembly()


    def init_solution(self):
        uh0 = self.mesh.interpolate(self.pde.init_solution, etype='node').reshape(-1)
        self.uh = uh0.copy()


    def step(self, n, tau):
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
        """
        Execute time-stepping on successively refined meshes.
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
        self.generate_mesh()
        prev_err_end = None
        
        for level in range(self.maxit):
            self._linear_system()
            self.init_solution()
            sols = [self.uh.copy()]
            em = bm.zeros((3, self.nt+1), dtype=float)
            em[:, 0] = self.mesh.error(lambda p: self.pde.solution(p, t0),self.uh,errortype='all')
            for n in range(1, self.nt+1):
                t_n = t0 + n * self.tau
                self.step(n, self.tau)
                sols.append(self.uh.copy())
                em[:, n] = self.mesh.error(
                    lambda p: self.pde.solution(p, t_n),
                    self.uh,
                    errortype='all'
                )
            self.all_solutions[level] = sols
            self.all_errors[level]    = em
            err_end = em[0, -1]  # max-norm at final time
            if prev_err_end is not None:
                ratio = prev_err_end / err_end
                print(f"refinement {level}:  final error={err_end:.2e}, ratio={ratio:.2f}")
            else:
                print(f"refinement {level}:  final error={err_end:.2e}")
            prev_err_end = err_end
            if level < self.maxit - 1:
                self.mesh.uniform_refine()

    def show_solution(self, zlim=None, interval=100):
        """
        Animate the 2D surface of u(x,y) over time for a given refinement level.
        Parameters:
        - level: refinement level key in self.all_solutions
        - tau: time step size (if None, uses stored self.tau)
        - zlim: tuple (zmin, zmax) for vertical axis limits
        - interval: delay between frames in ms
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        mesh = self.mesh
        sol = self.all_solutions[self.maxit - 1]  # Get the last refinement level solution
        # time step
        dt = self.tau if self.tau is not None else (self.pde.duration()[1] - self.t0) / len(sol)
        # prepare coords
        node = mesh.entity('node')
        x, y = node[:, 0], node[:, 1]
        tri = mesh.entity('cell')
        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if zlim:
            ax.set_zlim(*zlim)
        # initial plot
        surf = ax.plot_trisurf(x, y, sol[0], triangles=tri, cmap='viridis')
        def update(frame):
            # update only Z data by reassigning collection
            ax.clear()
            t = self.t0 + frame * dt
            ax.set_title(f"Time = {t:.2f}")
            ax.plot_trisurf(x, y, sol[frame], triangles=tri, cmap='viridis')
            return ax.collections
        ani = animation.FuncAnimation(
            fig, update, frames=len(sol), interval=interval, blit=False
        )
        plt.show()
    def show_error(self, level=0, error_type=0):
        """
        Plot a single error metric over time for a given refinement level.
        Parameters:
        - level: refinement level key in self.all_errors
        - error_type: index of error to plot (0:max, 1:L2, 2:H1)
        """
        import matplotlib.pyplot as plt
        em = self.all_errors[level]  # shape (3, nt+1)
        # time array
        nt = em.shape[1] - 1
        t0, t1 = self.pde.duration()
        times = [t0 + i*(t1 - t0)/nt for i in range(nt+1)]
        # select error sequence
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