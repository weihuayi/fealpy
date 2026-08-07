"""Manufactured-case adapter for the collocated SIMPLE Stokes solve."""

from fealpy.model import ComputationalModel, PDEModelManager

from .collocated_simple_solver import CollocatedSimpleSolver
from .cell_average_error import cell_average_l2_error
from .engineering_boundary_conditions import (
    PDEBoundaryConditions,
    resolve_simple_boundary_conditions,
)
from .collocated_linear_solvers import CollocatedNSLinearSolvers
from .solver_controls import positive_scalar
from .steady_ns_solver_profiles import (
    SteadyNSSimpleProfile,
    steady_ns_high_accuracy_simple_profile,
)


class StokesFVMSimpleModel(ComputationalModel):
    """Finite-volume SIMPLE model for Stokes PDE examples."""

    def __init__(self, options):
        self.options = options
        self._validate_options()
        ComputationalModel.__init__(
            self,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.pde = self._resolve_stokes_pde(options["pde"])
        self.error_quadrature_order = int(options.get("error_quadrature_order", 4))
        mesh = self._init_mesh(options)
        self.mu = self._init_diffusion_coef(options)
        profile = options.get("profile")
        if profile is None:
            profile = steady_ns_high_accuracy_simple_profile()
        if not isinstance(profile, SteadyNSSimpleProfile):
            raise TypeError("profile must be a SteadyNSSimpleProfile.")
        self.profile = profile
        boundary_conditions = resolve_simple_boundary_conditions(
            mesh,
            PDEBoundaryConditions(
                mesh,
                dirichlet_velocity=self.pde.dirichlet_velocity,
            ),
            profile.discretization,
            profile.pressure_system,
        )
        linear_solvers = options.get("linear_solvers")
        if linear_solvers is None:
            linear_solvers = profile.build_linear_solvers()
        if not isinstance(
            linear_solvers,
            CollocatedNSLinearSolvers,
        ):
            raise TypeError(
                "linear_solvers must be CollocatedNSLinearSolvers."
            )
        self.linear_solvers = linear_solvers
        self.solver = CollocatedSimpleSolver(
            diffusion_coef=self.mu,
            convection_coef=0.0,
            source=self.pde.source,
            boundary_conditions=boundary_conditions,
            discretization_controls=profile.discretization,
            iteration_controls=profile.iteration,
            linear_solvers=linear_solvers,
            logger=self.logger,
        )
        self.mesh = mesh
        self.fvm_geometry = boundary_conditions.physical.geometry
        self.NC = self.fvm_geometry.NC
        self.GD = self.fvm_geometry.GD

    def _validate_options(self) -> None:
        allowed = {
            "pde",
            "mesh_type",
            "mesh_refine",
            "nx",
            "ny",
            "nz",
            "mu",
            "error_quadrature_order",
            "profile",
            "linear_solvers",
            "pbar_log",
            "log_level",
        }
        unsupported = set(self.options).difference(allowed)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported StokesFVMSimpleModel options: {names}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.NC} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def close(self) -> None:
        """Release third-party linear-solver resources owned by this model."""
        self.linear_solvers.close()

    @staticmethod
    def _resolve_stokes_pde(pde):
        return PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def _init_mesh(self, options):
        mesh_type = (
            options.get("mesh_type")
            or getattr(self.pde, "default_mesh_type", "uniform_quad")
        )
        mesh_type = {"quad": "uniform_quad", "tri": "uniform_tri"}.get(
            mesh_type, mesh_type
        )
        mesh_refine = int(options.get("mesh_refine", 0))
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")

        if getattr(self.pde, "supports_geometric_refine", False):
            return self.pde.init_mesh[mesh_type](mesh_refine=mesh_refine)

        mesh_options = {
            name: int(options[name])
            for name in ("nx", "ny", "nz")
            if options.get(name) is not None
        }
        mesh = self.pde.init_mesh[mesh_type](**mesh_options)
        if mesh_refine == 0:
            return mesh
        if hasattr(mesh, "uniform_refine"):
            for _ in range(mesh_refine):
                mesh.uniform_refine()
            return mesh
        raise ValueError("mesh does not provide uniform_refine().")

    def _init_diffusion_coef(self, options):
        if options.get("mu") is not None:
            return positive_scalar(options["mu"], "mu")
        for name in ("viscosity", "mu"):
            if hasattr(self.pde, name):
                return positive_scalar(getattr(self.pde, name), "mu")
        return 1.0

    def solve(self):
        """Run one cold-start SIMPLE solve and return its immutable result."""
        return self.solver.solve()

    def compute_error(self, result) -> tuple[float, ...]:
        """Compute errors against exact control-volume averages."""
        velocity_error, exact_velocity = cell_average_l2_error(
            self.mesh,
            self.pde.velocity,
            result.velocity,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        pressure_error, exact_pressure = cell_average_l2_error(
            self.mesh,
            self.pde.pressure,
            result.pressure,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        return tuple(velocity_error[i] for i in range(self.GD)) + (pressure_error,)

    def plot(self, result) -> None:
        """Plot numerical and exact solution errors for u, v, and p."""
        import matplotlib.pyplot as plt

        _, exact_velocity = cell_average_l2_error(
            self.mesh,
            self.pde.velocity,
            result.velocity,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        _, exact_pressure = cell_average_l2_error(
            self.mesh,
            self.pde.pressure,
            result.pressure,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        cell_centers = self.fvm_geometry.cell_center
        x, y = cell_centers[:, 0], cell_centers[:, 1]

        fig = plt.figure(figsize=(15, 10))
        titles = [
            ("Error u", result.velocity[:, 0] - exact_velocity[:, 0]),
            ("Error v", result.velocity[:, 1] - exact_velocity[:, 1]),
            ("Error p", result.pressure - exact_pressure),
        ]
        for i, (title, data) in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            ax.plot_trisurf(x, y, data, cmap="viridis")
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_residual(self, result) -> None:
        """Plot residual decay curve."""
        import matplotlib.pyplot as plt

        mass = [
            residual.mass_relative_l2
            for residual in result.residual_history
        ]
        pressure_correction = [
            residual.pressure_correction_l2
            for residual in result.residual_history
        ]
        plt.figure(figsize=(8, 5))
        plt.semilogy(mass, marker="o", linestyle="-", color="b", label="mass")
        plt.semilogy(
            pressure_correction,
            marker="s",
            linestyle="-",
            color="r",
            label="pressure correction",
        )
        plt.legend()
        plt.title("SIMPLE Residuals vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
