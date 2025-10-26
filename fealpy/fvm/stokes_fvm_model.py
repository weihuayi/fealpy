from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm

from fealpy.solver import spsolve

from ..fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DivIntegrator,
    GradientReconstruct,
    DirichletBC,
)

class StokesFVMModel(ComputationalModel):
    """
    The Stokes equation in two-dimensional cases is solved by the finite volume method
    using the SIMPLE algorithm. The velocity and pressure are iteratively corrected
    to satisfy the continuity equation.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "WARNING"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options["space_degree"])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[str, object]) -> None:
        """Set the PDE model."""
        self.pde = PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        """Set the computational mesh."""
        self.mesh = self.pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)
        self.cm = self.mesh.entity_measure('cell')
        self.cmv = bm.tile(self.cm, 2)

    def set_space(self, degree: int = 0) -> None:
        """Set the function spaces for velocity and pressure."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))
        self.NC = self.mesh.number_of_cells()

    def temporary_velocity(self, p) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        B = bform.assembly()

        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)
        A = B.diags()

        grad_p = GradientReconstruct(self.mesh).AverageGradientreconstruct(p)
        px = bm.transpose(grad_p).flatten()
        pin = bm.einsum('i,i->i', px, self.cmv)
        f = f - pin

        u = spsolve(B, f)
        return A, u

    def pressure_correct(self, A: TensorLike, u: TensorLike) -> TensorLike:
        """Solve for pressure correction p' to enforce continuity."""
        bform3 = BilinearForm(self.velocity_space)
        bform3.add_integrator(DivIntegrator(q=self.p + 2))
        D = bform3.assembly()
        div_u = D @ u

        F = bm.zeros(2 * self.NC)
        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        F = dbc.DivApply(F)
        div_u = div_u - F
        div_u = A @ div_u
        div_u = div_u[:self.NC] + div_u[self.NC:]
        
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        C = bform2.assembly()
        dbc2 = DirichletBC(self.mesh, self.pde.dirichlet_pressure)
        C, div_u = dbc2.DiffusionApply(C, div_u)

        p_c = spsolve(C, div_u)
        return p_c

    def solve(self, max_iter: int = 10, tol: float = 1e-5) -> Tuple[TensorLike, TensorLike]:
        """Solve the Stokes equation using the SIMPLE algorithm."""
        p = bm.zeros(self.NC)
        A, u = self.temporary_velocity(p)
        self.residuals = []
        for i in range(max_iter):
            p_c = self.pressure_correct(A, u)
            L2p_c = bm.sqrt(bm.sum(self.cm * (p_c)**2))
            self.residuals.append(float(L2p_c))
            self.logger.info(f"[Iter {i+1}] pressure_correct = {L2p_c:.4e}")
            if L2p_c < tol:
                self.logger.info("Converged.")
                break
            p = p - p_c
            A, u = self.temporary_velocity(p)

        self.uh = u
        self.ph = p
        return u, p

    def compute_error(self) -> Tuple[float, float]:
        """Compute L2 errors for velocity and pressure."""
        cell_centers = self.mesh.entity_barycenter('cell')
        u_exact = self.pde.velocity(cell_centers)
        u_exact = bm.transpose(u_exact).flatten()
        p_exact = self.pde.pressure(cell_centers)
        Verror = bm.sqrt(bm.sum(self.cmv * (self.uh - u_exact)**2))
        Perror = bm.sqrt(bm.sum(self.cm * (self.ph - p_exact)**2))
        return Verror, Perror

    def plot(self) -> None:
        """Plot numerical and exact solutions for u1, u2, and p."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        cell_centers = self.mesh.entity_barycenter('cell')
        x, y = cell_centers[:, 0], cell_centers[:, 1]
        u1, u2 = self.uh[:self.NC], self.uh[self.NC:]
        u_exact = self.pde.velocity(cell_centers)
        p_exact = self.pde.pressure(cell_centers)

        fig = plt.figure(figsize=(15, 10))
        titles = [
            ("Numerical u1", u1), ("Exact u1", u_exact[:, 0]),
            ("Numerical u2", u2), ("Exact u2", u_exact[:, 1]),
            ("Numerical p", self.ph), ("Exact p", p_exact),
        ]
        for i, (title, data) in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            ax.plot_trisurf(x, y, data, cmap='viridis')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_residual(self) -> None:

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.residuals, marker='o', linestyle='-', color='b')
        plt.title("Pressure Correction Residual vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()