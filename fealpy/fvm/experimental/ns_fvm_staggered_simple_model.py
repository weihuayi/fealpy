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
    ConvectionIntegrator,
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

class NSFVMStaggeredSimpleModel(ComputationalModel):
    """
    A 2D Navier-Stokes solver using finite volume method on staggered mesh.
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
        self.pde = PDEModelManager("navier_stokes").get_example(pde) if isinstance(pde, int) else pde

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
        self.u_gradient = GradientReconstruct(self.umesh)
        self.v_gradient = GradientReconstruct(self.vmesh)

    def compute_temporary_velocity_u(self, p_u, uf) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        uspace = ScaledMonomialSpace2d(self.umesh, 0)

        bform = BilinearForm(uspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2, coef=uf))
        A = bform.assembly()

        f = LinearForm(uspace).add_integrator(ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        grad_p = self.u_gradient.cell_gradient(p_u)
        f -= bm.einsum('i,i->i', grad_p[:, 0], self.ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        A, f = dbc.apply_diffusion(A, f, components=1)
        A, f = dbc.ThresholdApply(A, f)
        uap = A.diags().values
        return spsolve(A, f,"mumps"), uap

    def compute_temporary_velocity_v(self, p_v, uf) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)

        bform = BilinearForm(vspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2, coef=uf))
        A = bform.assembly()

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
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        # Mathematical risk:
        # This legacy SIMPLE coefficient uses face measure / a_p_edge.  It is
        # not the same response used by the cleaned staggered PISO model
        # (velocity control-volume response V/a_p mapped to pressure faces).
        # Keep it unchanged for the current SIMPLE baseline, but do not treat
        # it as a validated general pressure-correction coefficient.
        p_edge = self.pmesh.entity_measure('edge')
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2,coef=p_edge / a_p_edge)
        ).assembly()  
        LagA = self.pmesh.entity_measure('cell')
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
        sol = spsolve(A, b)
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
        relax: float = 0.32,
        tol_mass=None,
        tol_pressure_update=None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """
        Solve the Navier-Stokes equation using the SIMPLE algorithm.
        """
        if relax <= 0:
            raise ValueError("relax must be positive.")

        tol_mass = tol if tol_mass is None else tol_mass
        tol_pressure_update = 10.0 * tol if tol_pressure_update is None else tol_pressure_update
        pressure_relax = relax
        field_dtype = self.pcm.dtype
        p = bm.zeros(self.ppoints.shape[0], dtype=field_dtype)
        UNE = self.umesh.number_of_edges()
        VNE = self.vmesh.number_of_edges()
        uf = bm.zeros(UNE, dtype=field_dtype)
        vf = bm.zeros(VNE, dtype=field_dtype)
        self.residuals = []

        vf_umesh = self.staggered_mesh.map_v_to_u_edges(
            vf, self.pde.dirichlet_velocity
        )
        Uf = bm.stack([uf, vf_umesh], axis=1)

        uf_vmesh = self.staggered_mesh.map_u_to_v_edges(
            uf, self.pde.dirichlet_velocity
        )
        Vf = bm.stack([uf_vmesh, vf], axis=1)
        ue2c = self.umesh.edge_to_cell()
        ve2c = self.vmesh.edge_to_cell()
        for i in range(max_iter):

            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p)
            uh, a_p_u = self.compute_temporary_velocity_u(p_u,Uf)
            vh, a_p_v = self.compute_temporary_velocity_v(p_v,Vf)
            
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
            ucell2pedge = self.staggered_mesh.get_dof_mapping_ucell2pedge()
            vcell2pedge = self.staggered_mesh.get_dof_mapping_vcell2pedge()
            pe2c = self.pmesh.edge_to_cell()
            u_corr = (p_corr[pe2c[ucell2pedge,0]]-p_corr[pe2c[ucell2pedge,1]])/self.staggered_mesh.hx
            v_corr = (p_corr[pe2c[vcell2pedge,0]]-p_corr[pe2c[vcell2pedge,1]])/self.staggered_mesh.hy
            u_corr = self.ucm / a_p_u * u_corr
            v_corr = self.vcm / a_p_v * v_corr
            p += p_update
            uh += u_corr
            vh += v_corr
            uf1 = (uh[ue2c[:,0]] + uh[ue2c[:,1]])/2
            vf1 = (vh[ve2c[:,0]] + vh[ve2c[:,1]])/2
            vf_umesh = self.staggered_mesh.map_v_to_u_edges(
                vf1, self.pde.dirichlet_velocity
            )
            Uf = bm.stack([uf1, vf_umesh], axis=1)
            uf_vmesh = self.staggered_mesh.map_u_to_v_edges(
                uf1, self.pde.dirichlet_velocity
            )
            Vf = bm.stack([uf_vmesh, vf1], axis=1)

        
        self.uh, self.vh, self.ph = uh, vh, p
        self.edge_vel, _ = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
        self.p_correct = p_corr
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute L2 errors for velocity and pressure.
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
        ax1.plot_trisurf(px, py, self.ph-self.pI, cmap="viridis")
        ax1.set_title("Pressure")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.plot_trisurf(ux, uy, self.uh-self.uI, cmap="viridis")
        ax2.set_title("U velocity")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot_trisurf(vx, vy, self.vh-self.vI, cmap="viridis")
        ax3.set_title("V velocity")

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

    def plot_streamline(self) -> None:
        """Plot the streamlines of the velocity field."""
        import matplotlib.pyplot as plt

        nx, ny = 32, 32 # u.shape = (ny, nx) 通常
        
        c2e = self.pmesh.cell2edge
        u = (self.edge_vel[c2e[:,1]]+self.edge_vel[c2e[:,3]])/2
        v = (self.edge_vel[c2e[:,0]]+self.edge_vel[c2e[:,2]])/2
        u2d = u.reshape((ny, nx),order="F")
        v2d = v.reshape((ny, nx),order="F")
        p2d = self.ph.reshape((ny, nx),order="F")

        import numpy as np

        dx = 1.0 / nx
        dy = 1.0 / ny

        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        X, Y = np.meshgrid(x, y)


        import matplotlib.pyplot as plt

        speed = np.sqrt(u2d**2 + v2d**2)

        plt.figure(figsize=(6, 6))
        plt.streamplot(
            X, Y, u2d, v2d,
            color=speed,
            cmap="viridis",
            density=1.5
        )
        plt.colorbar(label="|u|")
        plt.axis("equal")
        plt.title("Lid-driven cavity streamlines")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, p2d, levels=50, cmap="coolwarm")
        plt.colorbar(label="p")
        plt.axis("equal")
        plt.title("Pressure contour")
        plt.show()
