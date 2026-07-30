from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    cell_average_l2_error,
)
from .legacy_boundary_conditions import ExperimentalDirichletBC as DirichletBC
from .staggered_mesh_manager import StaggeredMeshManager
from .staggered_divergence_reconstruct import (
    StaggeredDivergenceReconstruct,
    staggered_mass_residual,
)
from ..simple_residual import (
    cell_l2_norm,
    relative_l2_update,
)

class StokesFVMStaggeredSimpleModel(ComputationalModel):
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
        self.staggered_mesh = StaggeredMeshManager(self.pde.domain(), nx, ny)
        self.umesh = self.staggered_mesh.umesh
        self.vmesh = self.staggered_mesh.vmesh
        self.pmesh = self.staggered_mesh.pmesh
        self.div = StaggeredDivergenceReconstruct(self.pmesh)
        self.pcm = self.pmesh.entity_measure("cell")
        self.ucm = self.umesh.entity_measure("cell")
        self.vcm = self.vmesh.entity_measure("cell")
        self.ppoints = self.pmesh.entity_barycenter("cell")
        self.upoints = self.umesh.entity_barycenter("cell")
        self.vpoints = self.vmesh.entity_barycenter("cell")
        self.u_gradient = GradientReconstruct(self.umesh, method="green_gauss")
        self.v_gradient = GradientReconstruct(self.vmesh, method="green_gauss")

    def compute_velocity_u(self, p_u) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        uspace = ScaledMonomialSpace2d(self.umesh, 0)
        A = BilinearForm(uspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(uspace).add_integrator(ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        grad_p = self.u_gradient.cell_gradient(p_u)
        f -= bm.einsum('i,i->i', grad_p[:, 0], self.ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        A, f = dbc.apply_diffusion(A, f, components=1)
        A, f = dbc.ThresholdApply(A, f)
        uap = A.diags().values
        return spsolve(A, f,"mumps"), uap

    def compute_velocity_v(self, p_v) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)
        A = BilinearForm(vspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(vspace).add_integrator(ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        grad_p = self.v_gradient.cell_gradient(p_v)
        f -= bm.einsum('i,i->i', grad_p[:, 1], self.vcm)
        dbc = DirichletBC(self.vmesh, self.pde.dirichlet_velocity_v,
                          threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
        A, f = dbc.apply_diffusion(A, f, components=1)
        A, f = dbc.ThresholdApply(A, f)
        vap = A.diags().values
        return spsolve(A, f,"mumps"), vap

    def correct_pressure_compute(self, f: TensorLike, a_p_edge: TensorLike) -> TensorLike:
        """
        Solve for pressure correction p' to enforce continuity.
        """
        LagA = self.pmesh.entity_measure('cell')
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        # Mathematical risk:
        # This is a historical pressure-correction coefficient.  The
        # edge_length**2/a_p_edge scaling should be re-derived before this
        # Stokes SIMPLE model is used as a reference implementation.
        p_edge = self.pmesh.entity_measure('edge')
        p_edge2 = bm.einsum('i,i->i', p_edge,p_edge)
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2,coef=p_edge2 / a_p_edge)
        ).assembly()
        # LagA = self.pmesh.entity_measure('cell')
        gauge_index = bm.stack(
            [
                bm.zeros(len(LagA), dtype=bm.int32),
                bm.arange(len(LagA), dtype=bm.int32),
            ],
            axis=0,
        )
        A1 = COOTensor(gauge_index, LagA, spshape=(1, len(LagA)))
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b = bm.concatenate([f, b0], axis=0)
        sol = spsolve(A, b,"mumps")
        p_correct = sol[:-1]   
        return p_correct

    def _simple_iteration_log_message(
        self,
        *,
        simple_iteration: int,
        nonorthogonal_iterations: int,
        pressure_criterion: float,
        pressure_relax: float,
        mass_residual: float,
        pressure_correction: float,
    ) -> str:
        """Format the main SIMPLE iteration diagnostic line."""
        return (
            f"[SIMPLE {simple_iteration}] "
            f"nonorthogonal iterations: {nonorthogonal_iterations}, "
            f"pressure criterion: {pressure_criterion:.2e}, "
            f"pressure relax: {pressure_relax:.2e}, "
            f"mass residual: {mass_residual:.2e}, "
            f"pressure correction L2: {pressure_correction:.2e}"
        )

    def solve(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        relax: float = 0.02,
        tol_mass=None,
        tol_pressure_update=None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Solve the Stokes equation using the SIMPLE algorithm."""
        if relax <= 0:
            raise ValueError("relax must be positive.")

        tol_mass = tol if tol_mass is None else tol_mass
        tol_pressure_update = 10.0 * tol if tol_pressure_update is None else tol_pressure_update
        pressure_relax = relax
        p = bm.zeros(self.ppoints.shape[0], dtype=self.pcm.dtype)
        self.residuals = []
        for i in range(max_iter):
            
            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p)
            uh, a_p_u = self.compute_velocity_u(p_u)
            vh, a_p_v = self.compute_velocity_v(p_v)
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
            self.div_rhs = self.div.StagReconstruct(edge_vel)
            p_corr = self.correct_pressure_compute(-self.div_rhs, a_p_edge)
            p_update = pressure_relax * p_corr
            residual = {
                "mass": staggered_mass_residual(
                    self.div.geometry,
                    edge_vel,
                ),
                "pressure_update": relative_l2_update(
                    p_update,
                    p,
                    geometry=self.div.geometry,
                ),
                "pressure_correction": cell_l2_norm(
                    p_corr,
                    geometry=self.div.geometry,
                ),
            }
            self.residuals.append(residual)
            residual["pressure_relax"] = pressure_relax
            residual["pressure_relax_reduced"] = False
            residual["pressure_relax_action"] = "fixed"
            residual["nonorthogonal_iterations"] = 0
            self.logger.info(
                self._simple_iteration_log_message(
                    simple_iteration=i + 1,
                    nonorthogonal_iterations=residual["nonorthogonal_iterations"],
                    pressure_criterion=residual["pressure_update"],
                    pressure_relax=pressure_relax,
                    mass_residual=residual["mass"],
                    pressure_correction=residual["pressure_correction"],
                )
            )
            if (
                residual["mass"] < tol_mass
                and residual["pressure_update"] < tol_pressure_update
            ):
                self.logger.info("Converged.")
                break
            p += p_update
        self.uh, self.vh, self.ph = uh, vh, p
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute errors for velocity and pressure.
        """
        q = getattr(self, "error_quadrature_order", 4)
        uerror, self.uI = cell_average_l2_error(self.umesh, self.pde.velocity_u, self.uh, q=q)
        verror, self.vI = cell_average_l2_error(self.vmesh, self.pde.velocity_v, self.vh, q=q)
        perror, self.pI = cell_average_l2_error(self.pmesh, self.pde.pressure, self.ph, q=q)
        return uerror, verror, perror

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 5))

        px, py = self.pmesh.entity_barycenter("cell").T
        ux, uy = self.umesh.entity_barycenter("cell").T
        vx, vy = self.vmesh.entity_barycenter("cell").T

        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.plot_trisurf(ux, uy, self.uh-self.uI, cmap="viridis")
        ax1.set_title("Error u")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.plot_trisurf(vx, vy, self.vh-self.vI, cmap="viridis")
        ax2.set_title("Error v")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot_trisurf(px, py, self.ph-self.pI, cmap="viridis")
        ax3.set_title("Error p")

        plt.tight_layout()
        plt.show()

    def plot_residual(self) -> None:
        """Plot residual decay curve."""
        import matplotlib.pyplot as plt
        mass = [residual["mass"] for residual in self.residuals]
        pressure_update = [
            residual["pressure_update"] for residual in self.residuals
        ]
        plt.figure(figsize=(8, 5))
        plt.semilogy(mass, marker="o", linestyle="-", color="b", label="mass")
        plt.semilogy(
            pressure_update,
            marker="s",
            linestyle="-",
            color="r",
            label="pressure update",
        )
        plt.legend()
        plt.title("SIMPLE Residuals vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
