from ..backend import backend_manager as bm
import matplotlib.pyplot as plt
from ..mesh import UniformMesh
from ..model import ComputationalModel
from ..model import PDEDataManager
from ..solver import spsolve
from . import LaplaceOperator, DirichletBC



class PoissonFDMModel(ComputationalModel):
    """Finite Difference Method solver for Poisson equation
    
    This class implements a finite difference solver for Poisson equations with
    uniform mesh refinement capabilities. It supports error analysis and solution
    visualization for 1D and 2D problems.
    
    Parameters
    ----------
        example : str, optional, default='sinsin'
            Name of the example problem to solve
        maxit : int, optional, default=4
            Maximum number of mesh refinement iterations
        ns : int, optional, default=20
            Initial number of segments per dimension
        solver : callable, optional, default=spsolve
            Linear system solver function (e.g., scipy.sparse.linalg.spsolve)
        
    Attributes
    ----------
        pde : PDEDataManager
            Manages PDE problem data (domain, solution, source term, etc.)
        maxit : int
            Maximum refinement iterations
        ns : int
            Initial segments per dimension
        solver : callable
            Linear system solver function
        mesh : UniformMesh
            Current computational mesh
        uh : ndarray
            Numerical solution vector
        em : ndarray
            Error matrix storing L∞, L2 and H1 errors for each refinement
    """

    def __init__(self, example: str = 'sinsin', maxit: int = 4, ns: int = 20, solver=spsolve):
        """Initialize Poisson FDM solver with given parameters"""
        self.pde = PDEDataManager('poisson').get_example(example) 
        self.maxit = maxit
        self.ns = ns
        self.solver = spsolve


    def run(self):
        """Execute the solver with mesh refinement
        
        Performs the following steps:
            1. Generates initial mesh
            2. For each refinement level:
            - Assembles linear system
            - Solves the system
            - Computes errors (max, L2, H1)
            - Refines mesh (except final iteration)
            3. Prints final errors and error ratios
        """
        maxit = self.maxit
        self.em = bm.zeros((3, maxit), dtype=bm.float64)
        self._generate_mesh(n=self.ns)

        for i in range(maxit):
            A, f = self._linear_system()
            self.uh = self.solver(A, f)
            self.em[0, i], self.em[1, i], self.em[2, i] = self.mesh.error(
                self.pde.solution, self.uh, errortype='all')
            if i < maxit-1:
                self.mesh.uniform_refine() 

        print("误差(max, L2, h1\l2): ", self.em, "收敛阶: ", self.em[:, 0:-1] / self.em[:, 1:], sep='\n')
    

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
        """Assemble the linear system Ax = f
        
        Returns
        -------
        A : sparse matrix
            Finite difference stiffness matrix
        f : ndarray
            Right-hand side vector after applying boundary conditions
        """
        pde = self.pde
        mesh = self.mesh
        A = LaplaceOperator(mesh=mesh).assembly()

        node = mesh.entity("node")
        f = pde.source(node)  

        A, f = DirichletBC(mesh=mesh, gd=pde.dirichlet).apply(A,f)
        return A, f


    def show_error(self):
        """Visualize error convergence across refinement levels
        
        Creates a 2-panel figure showing:
        - Left: Error norms (max, L2, H1\l2) vs refinement level
        - Right: Error ratios vs refinement level with reference line at y=4
        """
        if self.pde.geo_dimension() == 1:
            error_names = ['max', 'L2', 'H1']
        else:
            error_names = ['max', 'L2', 'l2']
        markers = ['o-', 's--', '^:']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  
        # Plot error curves (left panel)
        for i in range(3):
            ax1.plot(self.em[i, :], markers[i], label=error_names[i], linewidth=4) 
        ax1.set_xlabel('Refinement Level', fontsize=24) 
        ax1.legend(fontsize=20, loc='upper right')  
        ax1.grid(True)
        ax1.set_title(' Error ', fontsize=28)  

        # Plot error ratios (right panel)
        em_ratio = self.em[:, 0:-1] / self.em[:, 1:]
        for i in range(3):
            ax2.plot(em_ratio[i, :], markers[i], label=f'{error_names[i]} ratio', linewidth=2)
        ax2.axhline(y=4, color='r', linestyle='-', label='y=4 (expected convergence order)', linewidth=2)
        ax2.set_xlabel('Refinement Level', fontsize=24)
        ax2.legend(fontsize=16, loc='upper right')
        ax2.grid(True)
        ax2.set_title('Convergence order', fontsize=28)


    def show_solution(self):
        """Display numerical solution visualization
        
        Note
        ----
        Only supports 1D and 2D problems. For 1D: plots solution curve.
        For 2D: shows 3D surface plot. Higher dimensions will print warning.
        """
        GD = self.pde.geo_dimension()
        
        if GD > 2:
            print("Warning: Only 1D and 2D function visualization is supported. Current problem dimension is ", GD)
            return
        
        fig = plt.figure()

        if GD == 2:
            axes = fig.add_subplot(111, projection='3d')
        elif GD == 1:
            axes = fig.add_subplot()

        self.mesh.show_function(axes, self.uh)
        plt.title(f"The numerical solution of the final mesh refinement")