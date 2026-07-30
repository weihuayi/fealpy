"""Cell-centred finite-volume model for scalar Poisson problems."""

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.functionspace import ScaledMonomialSpace
from fealpy.fem import BilinearForm, LinearForm

from .cell_average_error import cell_average_l2_error
from .dirichlet_bc import DirichletBC
from .face_gradient import reconstruct_face_gradient
from .fvm_geometry import FVMGeometry
from .fvm_linear_solver import FVMLinearSolver
from .gradient_reconstruct import (
    GradientReconstruct,
    ResolvedGradientBoundary,
)
from .scalar_cross_diffusion_integrator import ScalarCrossDiffusionIntegrator
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator
from .solver_controls import PoissonSolverControls
from .solver_diagnostics import (
    equation_residual_converged,
    normalized_equation_residual,
)


class PoissonFVMModel(ComputationalModel):
    """Solve a scalar Poisson problem with deferred non-orthogonal correction."""

    def __init__(self, options):
        self.options = options
        self._validate_options()
        super().__init__(
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        pde_input = options["pde"]
        self.pde = (
            PDEModelManager("poisson").get_example(pde_input)
            if isinstance(pde_input, int)
            else pde_input
        )
        self.logger.info(self.pde)
        self.controls = PoissonSolverControls.from_mapping(options)
        self.error_quadrature_order = int(options.get("error_quadrature_order", 4))

        mesh_type = (
            options.get("mesh_type")
            or getattr(self.pde, "default_mesh_type", "uniform_tri")
        )
        mesh_refine = int(options.get("mesh_refine", 0) or 0)
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")
        if getattr(self.pde, "supports_geometric_refine", False):
            self.mesh = self.pde.init_mesh[mesh_type](mesh_refine=mesh_refine)
        else:
            mesh_options = {
                name: int(options[name])
                for name in ("nx", "ny", "nz")
                if options.get(name) is not None
            }
            self.mesh = self.pde.init_mesh[mesh_type](**mesh_options)
            if mesh_refine > 0:
                if not hasattr(self.mesh, "uniform_refine"):
                    raise ValueError("mesh does not provide uniform_refine().")
                for _ in range(mesh_refine):
                    self.mesh.uniform_refine()

        self.p = self.controls.space_degree
        self.space = ScaledMonomialSpace(self.mesh, self.p)
        self.fvm_geometry = FVMGeometry(self.mesh)
        self.cell_measure = self.fvm_geometry.cell_measure
        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        boundary_values = bm.array(
            self.pde.dirichlet(
                self.fvm_geometry.face_center[boundary_faces]
            ),
            dtype=self.fvm_geometry.cell_center.dtype,
            device=bm.get_device(self.fvm_geometry.cell_center),
        )
        empty_faces = boundary_faces[:0]
        self.gradient_boundary = ResolvedGradientBoundary(
            dirichlet_faces=boundary_faces,
            dirichlet_values=boundary_values,
            neumann_faces=empty_faces,
            neumann_sn_grad=bm.zeros(
                0,
                dtype=self.fvm_geometry.cell_center.dtype,
                device=bm.get_device(self.fvm_geometry.cell_center),
            ),
        )
        self.face_gradient_boundary = ResolvedGradientBoundary.empty(
            self.fvm_geometry,
        )
        self.gradient = GradientReconstruct(
            self.fvm_geometry,
            self.gradient_boundary,
            method=self.controls.gradient_method,
            layer_weights=self.controls.gradient_layer_weights,
            boundary_weight=self.controls.gradient_boundary_weight,
        )
        self.dirichlet_boundary = DirichletBC(
            self.fvm_geometry,
            boundary_faces,
            boundary_values,
            diffusion_method=self.controls.diffusion_method,
            nonorthogonal_eps=self.controls.diffusion_nonorthogonal_eps,
        )
        self.linear_solver = options.get("linear_solver")
        if self.linear_solver is None:
            self.linear_solver = FVMLinearSolver("scipy")

    def _validate_options(self) -> None:
        allowed = set(PoissonSolverControls.option_names()) | {
            "pde",
            "mesh_type",
            "mesh_refine",
            "nx",
            "ny",
            "nz",
            "error_quadrature_order",
            "linear_solver",
            "pbar_log",
            "log_level",
        }
        unsupported = set(self.options).difference(allowed)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported PoissonFVMModel options: {names}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh: {self.fvm_geometry.NC} cells\n"
            f"  Space degree: {self.p}\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def assemble_base_system(self) -> tuple:
        """Assemble the implicit two-point diffusion system."""
        bform = BilinearForm(self.space)
        bform.add_integrator(
            ScalarDiffusionIntegrator(
                coef=1,
                geometry=self.fvm_geometry,
                method=self.controls.diffusion_method,
                nonorthogonal_eps=self.controls.diffusion_nonorthogonal_eps,
            )
        )
        matrix = bform.assembly()
        lform = LinearForm(self.space)
        lform.add_integrator(
            ScalarSourceIntegrator(
                self.pde.source,
                q=2,
                geometry=self.fvm_geometry,
            )
        )
        rhs = lform.assembly()
        return self.dirichlet_boundary.apply_diffusion(
            matrix,
            rhs,
            components=1,
        )

    def compute_cross_diffusion(self, cell_values) -> TensorLike:
        """Assemble the explicit non-orthogonal diffusion correction."""
        gradient = self.gradient.cell_gradient(cell_values)
        face_gradient = reconstruct_face_gradient(
            self.fvm_geometry,
            gradient,
            cell_values,
            interpolation_method="average",
            boundary=self.face_gradient_boundary,
        )
        lform = LinearForm(self.space)
        lform.add_integrator(
            ScalarCrossDiffusionIntegrator(
                cell_values,
                face_gradient,
                coef=1,
                geometry=self.fvm_geometry,
                method=self.controls.diffusion_method,
                boundary_policy="all",
                cross_flux_limiter=self.controls.cross_flux_limiter,
                limit_coeff=self.controls.cross_flux_limit_coeff,
                nonorthogonal_eps=self.controls.diffusion_nonorthogonal_eps,
            )
        )
        return lform.assembly()

    def diffusion_residual(self, matrix, rhs, solution):
        """Return the full corrected diffusion residual and explicit RHS."""
        cross = self.compute_cross_diffusion(solution)
        return matrix @ solution - rhs - cross, cross

    def solve(self) -> TensorLike:
        """Solve the full deferred-correction equation to configured tolerance."""
        controls = self.controls
        matrix, rhs = self.assemble_base_system()
        solution = self.linear_solver.solve(matrix, rhs).solution
        initial_residual = None
        relative_update = 0.0

        for iteration in range(controls.nonorthogonal_max_iter + 1):
            _, cross = self.diffusion_residual(matrix, rhs, solution)
            lhs = matrix @ solution
            corrected_rhs = rhs + cross
            metrics = normalized_equation_residual(lhs, corrected_rhs)
            absolute = metrics.absolute
            relative = metrics.relative
            if initial_residual is None:
                initial_residual = absolute

            self.logger.info(
                "[NonOrth %d] absolute = %.4e, relative = %.4e",
                iteration,
                absolute,
                relative,
            )
            if equation_residual_converged(
                metrics,
                rtol=controls.nonorthogonal_rtol,
                atol=controls.nonorthogonal_atol,
            ):
                self.solution = solution
                self.nonorthogonal_diagnostics = {
                    "converged": True,
                    "iterations": iteration,
                    "initial_residual": initial_residual,
                    "final_residual": absolute,
                    "relative_residual": relative,
                    "relative_update": relative_update,
                    "relaxation": controls.nonorthogonal_relaxation,
                    "reached_max_iter": False,
                }
                return solution

            if iteration == controls.nonorthogonal_max_iter:
                break

            trial = self.linear_solver.solve(
                matrix,
                corrected_rhs,
            ).solution
            relaxation = controls.nonorthogonal_relaxation
            next_solution = (1.0 - relaxation) * solution + relaxation * trial
            update_norm = float(
                bm.to_numpy(bm.linalg.norm(next_solution - solution))
            )
            solution_norm = float(bm.to_numpy(bm.linalg.norm(next_solution)))
            relative_update = update_norm / max(solution_norm, 1.0e-30)
            solution = next_solution

        self.nonorthogonal_diagnostics = {
            "converged": False,
            "iterations": controls.nonorthogonal_max_iter,
            "initial_residual": initial_residual,
            "final_residual": absolute,
            "relative_residual": relative,
            "relative_update": relative_update,
            "relaxation": controls.nonorthogonal_relaxation,
            "reached_max_iter": True,
        }
        raise RuntimeError(
            "non-orthogonal correction did not converge before max_iter"
        )

    def compute_error(self) -> float:
        """Return the L2 error against the exact control-volume average."""
        self.error, self.exact_solution = cell_average_l2_error(
            self.mesh,
            self.pde.solution,
            self.solution,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        return self.error

    def plot(self) -> None:
        """Plot the numerical solution, exact cell average, and their error."""
        import matplotlib.pyplot as plt

        cell_center = self.fvm_geometry.cell_center
        x, y = cell_center[:, 0], cell_center[:, 1]
        fig = plt.figure(figsize=(10, 5))
        fields = (
            ("Numerical Solution (FVM)", self.solution, "viridis"),
            ("Exact Solution", self.exact_solution, "plasma"),
            ("Error (Exact - Numerical)", self.exact_solution - self.solution, "plasma"),
        )
        for index, (title, values, cmap) in enumerate(fields, start=1):
            axis = fig.add_subplot(1, 3, index, projection="3d")
            axis.plot_trisurf(x, y, values, cmap=cmap, linewidth=0.2)
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
        plt.tight_layout()
        plt.show()
