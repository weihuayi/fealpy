from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.fem import BilinearForm, LinearForm

from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    StaggeredMeshManager,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
)


class StokesFVMStaggeredModel(ComputationalModel):
    """
    A 2D Stokes solver using finite volume method on staggered mesh.
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "INFO"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.pmesh.number_of_cells()} pressure cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[str, object]) -> None:
        """Set the PDE model."""
        self.pde = PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        """Set the computational staggered mesh."""
        self.staggered_mesh = StaggeredMeshManager(self.pde, nx, ny)
        self.umesh = self.staggered_mesh.umesh
        self.vmesh = self.staggered_mesh.vmesh
        self.pmesh = self.staggered_mesh.pmesh
        self.div = DivergenceReconstruct(self.pmesh)
        self.cm = self.pmesh.entity_measure("cell")
        self.ppoints = self.pmesh.entity_barycenter("cell")
        self.upoints = self.umesh.entity_barycenter("cell")
        self.vpoints = self.vmesh.entity_barycenter("cell")

    def compute_velocity_u(self, p_u) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        uspace = ScaledMonomialSpace2d(self.umesh, 0)
        ucm = self.umesh.entity_measure("cell")
        A = BilinearForm(uspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(uspace).add_integrator(ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        grad_p = GradientReconstruct(self.umesh).AverageGradientreconstruct(p_u)
        f -= bm.einsum('i,i->i', grad_p[:, 0], ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        return spsolve(A, f), A.diags().values

    def compute_velocity_v(self, p_v) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)
        vcm = self.vmesh.entity_measure("cell")
        A = BilinearForm(vspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(vspace).add_integrator(ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        grad_p = GradientReconstruct(self.vmesh).AverageGradientreconstruct(p_v)
        f -= bm.einsum('i,i->i', grad_p[:, 1], vcm)
        dbc = DirichletBC(self.vmesh, self.pde.dirichlet_velocity_v,
                          threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        return spsolve(A, f), A.diags().values

    def compute_pressure(self, f: TensorLike, a_p_edge: TensorLike) -> TensorLike:
        """
        Solve for pressure correction p' to enforce continuity.
        """
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2, coef=1 / a_p_edge)
        ).assembly()
        dbc = DirichletBC(self.pmesh, self.pde.dirichlet_pressure)
        A, f = dbc.DiffusionApply(A, f)
        return spsolve(A, f)

    def solve(self, max_iter: int = 200, tol: float = 1e-6) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """
        Solve the Stokes equation using the SIMPLE algorithm.
        """
        p = bm.zeros(self.ppoints.shape[0])
        self.residuals = []
        for i in range(max_iter):
            p_u, p_v = self.staggered_mesh.interpolate_pressure_to_edges(p)
            uh, a_p_u = self.compute_velocity_u(p_u)
            vh, a_p_v = self.compute_velocity_v(p_v)
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_cell_to_edge(uh, vh, a_p_u, a_p_v)
            div_rhs = self.div.reconstruct(edge_vel, self.pde.velocity)
            p_corr = self.compute_pressure(-div_rhs, a_p_edge)
            err = bm.sqrt(bm.sum(self.cm * p_corr ** 2))
            self.residuals.append(float(err))
            self.logger.info(f"[Iter {i+1}] Pressure correction residual: {err:.2e}")
            if err < tol:
                self.logger.info("Converged.")
                break
            p += 0.8 * p_corr

        self.uh, self.vh, self.ph = uh, vh, p
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute L2 errors for velocity and pressure.
        """
        self.uI = self.pde.velocity_u(self.upoints)
        self.vI = self.pde.velocity_v(self.vpoints)
        self.pI = self.pde.pressure(self.ppoints)
        ue = bm.max(bm.abs(self.uh - self.uI))
        ve = bm.max(bm.abs(self.vh - self.vI))
        pe = bm.sqrt(bm.sum(self.cm * (self.ph - self.pI)**2))
        return ue, ve, pe

    def plot(self) -> None:
        """Plot numerical and exact solutions for u1, u2, and p."""
        import matplotlib.pyplot as plt
        x, y = self.ppoints[:, 0], self.ppoints[:, 1]

        fig = plt.figure(figsize=(12, 8))
        for i, (data, title) in enumerate([
            (self.uh, "Numerical u"),
            (self.vh, "Numerical v"), 
            (self.ph, "Numerical p"),
            (self.uI, "Exact u"),
            (self.vI, "Exact v"),
            (self.pI, "Exact p")
        ]):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            ax.plot_trisurf(x, y, data, cmap='viridis')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_residual(self) -> None:
        '''Plot the residual descent curve of pressure p correction.'''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.residuals, marker='o', linestyle='-', color='b')
        plt.title("Pressure Correction Residual vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
