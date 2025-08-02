from typing import Union

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.fem import BilinearForm, LinearForm

from fealpy.solver import spsolve

from ..fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarCrossDiffusionIntegrator,
    DirichletBC,
)


class PoissonFVMModel(ComputationalModel):
    """
    A 2D Poisson equation solver using the finite volume method (FVM).

    This model iteratively solves the diffusion problem with optional cross-diffusion,
    and supports structured quadrilateral meshes.

    Parameters:
        options (dict): Configuration dictionary for model setup.
            - 'pde': PDE data or index
            - 'nx', 'ny': mesh divisions
            - 'space_degree': polynomial degree of test space
            - 'pbar_log', 'log_level': logging controls

    Attributes:
        mesh : The initialized computational mesh.
        space : The finite volume function space.
        pde : The PDE model object.
        uh : Numerical solution vector (computed after calling `solve`).
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "WARNING"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options["space_degree"])

    def __str__(self) -> str:
        """Return a summary of the model configuration."""
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh: {self.mesh.number_of_cells()} cells\n"
            f"  Space degree: {self.p}\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[str, object]) -> None:
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde

        self.logger.info(self.pde)

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        self.mesh = self.pde.init_mesh['uniform_quad'](nx=nx, ny=ny)    

    def set_space(self, degree: int = 0) -> None:
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
    
    def assemble_base_system(self) -> tuple:
        """
        Assemble the base linear system A·u = f without nonlinear terms.

        Returns:
            A (spmatrix): The system matrix from diffusion term.
            f (ndarray): The source term vector.
        """
        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        A = bform.assembly()

        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet)
        A, f = dbc.DiffusionApply(A, f)
        return A, f

    def compute_cross_diffusion(self, uh) -> TensorLike:
        """
        Assemble the nonlinear cross-diffusion term.

        Parameters:
            uh (TensorLike): Current solution vector.

        Returns:
            ndarray: Right-hand side vector from cross-diffusion.
        """
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarCrossDiffusionIntegrator(uh, q=self.p + 2))
        return lform.assembly()

    def solve(self, max_iter=6, tol=1e-6) -> TensorLike:
        """
        Iteratively solve the linear system including cross-diffusion.

        Parameters:
            max_iter (int): Maximum number of fixed-point iterations.
            tol (float): Convergence tolerance for residual.

        Returns:
            uh (ndarray): The numerical solution.
        """
        A, f = self.assemble_base_system()
        uh = spsolve(A, f)

        for i in range(max_iter):
            cross = self.compute_cross_diffusion(uh)
            rhs = f + cross
            uh_new = spsolve(A, rhs)
            err = bm.max(bm.abs(uh_new - uh))

            self.logger.info(f"[Iter {i+1}] residual = {err:.4e}")
            if err < tol:
                self.logger.info("Converged.")
                break
            
            uh = uh_new

        self.uh = uh
        return uh

    def compute_error(self) -> float:
        """
        Compute the L2 error against the exact solution.

        Returns:
            float: The L2 norm of the error.
        """
        cell_center = self.mesh.entity_barycenter('cell')
        u_exact = self.pde.solution(cell_center)
        error = bm.sqrt(bm.sum(self.mesh.entity_measure('cell') * (u_exact - self.uh)**2))
        return error

    def plot(self) -> None:
        """
        Plot the numerical and exact solution using matplotlib.
        """
        import matplotlib.pyplot as plt
        cell_center = self.mesh.entity_barycenter('cell')  
        x, y = cell_center[:, 0], cell_center[:, 1]
        z_num = self.uh                     
        z_exact = self.pde.solution(cell_center)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_trisurf(x, y, z_num, cmap='viridis', linewidth=0.2)
        ax1.set_title("Numerical Solution (FVM)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u_h")
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_trisurf(x, y, z_exact, cmap='plasma', linewidth=0.2)
        ax2.set_title("Exact Solution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("u_exact")

        plt.tight_layout()
        plt.show()


