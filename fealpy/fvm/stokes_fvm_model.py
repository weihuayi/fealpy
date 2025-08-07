from typing import Union, Tuple
from ..backend import backend_manager as bm
from ..model import PDEModelManager, ComputationalModel
from ..functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from ..fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DivIntegrator,
    GradientReconstruct,
    DirichletBC,
)
from ..fem import BilinearForm, LinearForm
from ..solver import spsolve

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

    def set_pde(self, pde: Union[str, object]):
        """Set the PDE model."""
        if isinstance(pde, int):
            self.pde = PDEModelManager('stokes').get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, nx: int = 10, ny: int = 10):
        """Set the computational mesh."""
        self.mesh = self.pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)
        self.cm = self.mesh.entity_measure('cell')
        self.cmv = bm.tile(self.cm, 2)

    def set_space(self, degree: int = 0):
        """Set the function spaces for velocity and pressure."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))
        self.NC = self.mesh.number_of_cells()

    def temporary_velocity(self, p):
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

    def pressure_correct(self, A, u):
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

    def solve(self, max_iter: int = 1000, tol: float = 1e-5):
        """Solve the Stokes equation using the SIMPLE algorithm."""
        p = bm.zeros(self.NC)
        A, u = self.temporary_velocity(p)

        for i in range(max_iter):
            p_c = self.pressure_correct(A, u)
            Perror = bm.sqrt(bm.sum(self.cm * (p_c)**2))
            # self.logger.info(f"[Iter {i+1}] P_correct = {Perror:.4e}")
            if Perror < tol:
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

    def plot(self):
        """Plot numerical and exact solutions for u1, u2, and p."""
        import matplotlib.pyplot as plt
        cell_centers = self.mesh.entity_barycenter('cell')
        x, y = cell_centers[:, 0], cell_centers[:, 1]
        u1_num = self.uh[:self.NC]
        u2_num = self.uh[self.NC:]
        p_num = self.ph
        u_exact = self.pde.velocity(cell_centers)
        u1_exact, u2_exact = u_exact[:, 0], u_exact[:, 1]
        p_exact = self.pde.pressure(cell_centers)

        fig = plt.figure(figsize=(15, 10))

        # Plot u1
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot_trisurf(x, y, u1_num, cmap='viridis', linewidth=0.2)
        ax1.set_title("Numerical u1")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u1_h")

        ax2 = fig.add_subplot(2, 3, 4, projection='3d')
        ax2.plot_trisurf(x, y, u1_exact, cmap='plasma', linewidth=0.2)
        ax2.set_title("Exact u1")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("u1_exact")

        # Plot u2
        ax3 = fig.add_subplot(2, 3, 2, projection='3d')
        ax3.plot_trisurf(x, y, u2_num, cmap='viridis', linewidth=0.2)
        ax3.set_title("Numerical u2")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("u2_h")

        ax4 = fig.add_subplot(2, 3, 5, projection='3d')
        ax4.plot_trisurf(x, y, u2_exact, cmap='plasma', linewidth=0.2)
        ax4.set_title("Exact u2")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.set_zlabel("u2_exact")

        # Plot p
        ax5 = fig.add_subplot(2, 3, 3, projection='3d')
        ax5.plot_trisurf(x, y, p_num, cmap='viridis', linewidth=0.2)
        ax5.set_title("Numerical p")
        ax5.set_xlabel("x")
        ax5.set_ylabel("y")
        ax5.set_zlabel("p_h")

        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.plot_trisurf(x, y, p_exact, cmap='plasma', linewidth=0.2)
        ax6.set_title("Exact p")
        ax6.set_xlabel("x")
        ax6.set_ylabel("y")
        ax6.set_zlabel("p_exact")

        plt.tight_layout()
        plt.show()