"""Manufactured-case adapter for the collocated PISO solver."""

from typing import Any

from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel
from fealpy.model import PDEModelManager

from .collocated_piso_solver import (
    CollocatedPisoSolver,
    PisoCorrectorCallback,
    PisoSnapshotCallback,
)
from .collocated_pressure_system import (
    CollocatedPressureSystemControls,
)
from .collocated_linear_solvers import (
    CollocatedNSLinearSolvers,
    build_collocated_ns_linear_solvers,
)
from .cell_average_error import cell_average_l2_error
from .engineering_boundary_conditions import (
    EngineeringBoundaryConditions,
    PDEBoundaryConditions,
    resolve_piso_boundary_conditions,
)
from .piso_result import PisoSolveResult
from .solver_controls import PisoSolverControls, positive_scalar


class NSFVMPISOModel(ComputationalModel):
    """Finite-volume PISO model for PDE examples with exact solutions."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self._validate_options()
        ComputationalModel.__init__(
            self,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        pde_input = options["pde"]
        if isinstance(pde_input, int):
            pde = PDEModelManager("navier_stokes").get_example(pde_input)
        else:
            pde = pde_input
        self.pde = pde
        self.error_quadrature_order = int(options.get("error_quadrature_order", 4))

        rho_value = options.get("rho", None)
        if rho_value is None:
            rho_value = getattr(pde, "rho", 1.0)
        mu_value = options.get("mu", None)
        if mu_value is None:
            for name in ("mu", "viscosity", "nu"):
                if hasattr(pde, name):
                    mu_value = getattr(pde, name)
                    break
            else:
                mu_value = 1.0
        self.rho = positive_scalar(rho_value, "rho")
        self.mu = positive_scalar(mu_value, "mu")

        mesh_type = options.get("mesh_type", "uniform_quad")
        mesh_type = mesh_type or getattr(pde, "default_mesh_type", "uniform_quad")
        mesh_refine = int(options.get("mesh_refine", 0) or 0)
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")
        if getattr(pde, "supports_geometric_refine", False):
            mesh = pde.init_mesh[mesh_type](mesh_refine=mesh_refine)
        else:
            mesh_options = {}
            if options.get("nx") is not None:
                mesh_options["nx"] = int(options["nx"])
            if options.get("ny") is not None:
                mesh_options["ny"] = int(options["ny"])
            if options.get("nz") is not None:
                mesh_options["nz"] = int(options["nz"])
            mesh = pde.init_mesh[mesh_type](**mesh_options)
            if mesh_refine > 0:
                if not hasattr(mesh, "uniform_refine"):
                    raise ValueError("mesh does not provide uniform_refine().")
                for _ in range(mesh_refine):
                    mesh.uniform_refine()

        boundary_input = options.get("boundary_conditions")
        if boundary_input is None:
            boundary_input = PDEBoundaryConditions(
                mesh,
                dirichlet_velocity=pde.dirichlet_velocity,
            )
        elif callable(boundary_input):
            boundary_input = boundary_input(mesh, pde)

        if isinstance(boundary_input, EngineeringBoundaryConditions):
            self.engineering_bc = boundary_input
        elif isinstance(boundary_input, PDEBoundaryConditions):
            self.engineering_bc = None
        else:
            raise TypeError(
                "boundary_conditions must be PDEBoundaryConditions, "
                "EngineeringBoundaryConditions, or a factory(mesh, pde) "
                "returning one of these types."
            )
        controls = PisoSolverControls.from_mapping(options)
        pressure_system_controls = options.get(
            "pressure_system_controls"
        )
        if pressure_system_controls is None:
            pressure_system_controls = CollocatedPressureSystemControls()
        if not isinstance(
            pressure_system_controls,
            CollocatedPressureSystemControls,
        ):
            raise TypeError(
                "pressure_system_controls must be "
                "CollocatedPressureSystemControls."
            )
        boundary_conditions = resolve_piso_boundary_conditions(
            mesh,
            boundary_input,
            controls,
            pressure_system_controls,
        )
        linear_solvers = options.get("linear_solvers")
        if linear_solvers is None:
            linear_solvers = build_collocated_ns_linear_solvers()
        if not isinstance(
            linear_solvers,
            CollocatedNSLinearSolvers,
        ):
            raise TypeError(
                "linear_solvers must be "
                "CollocatedNSLinearSolvers."
            )
        self.linear_solvers = linear_solvers
        self.solver = CollocatedPisoSolver(
            diffusion_coef=self.mu,
            convection_coef=self.rho,
            source=pde.source,
            boundary_conditions=boundary_conditions,
            controls=controls,
            linear_solvers=linear_solvers,
        )
        self.mesh = mesh
        self.fvm_geometry = boundary_conditions.physical.geometry
        self.NC = self.fvm_geometry.NC
        self.GD = self.fvm_geometry.GD

    def close(self) -> None:
        """Release third-party linear-solver resources owned by this model."""
        self.linear_solvers.close()

    def _validate_options(self) -> None:
        allowed = set(PisoSolverControls.option_names()) | {
            "pde",
            "mesh_type",
            "mesh_refine",
            "nx",
            "ny",
            "nz",
            "rho",
            "mu",
            "error_quadrature_order",
            "boundary_conditions",
            "linear_solvers",
            "pressure_system_controls",
            "pbar_log",
            "log_level",
        }
        unsupported = set(self.options).difference(allowed)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported NSFVMPISOModel options: {names}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.NC} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
            f"  Time steps: {self.solver.controls.time_steps}\n"
            f"  PISO correctors: {self.solver.controls.n_correctors}\n"
            f"  Momentum nonorthogonal corrections: "
            f"{self.solver.controls.momentum_nonorthogonal_max_iterations}\n"
            f"  Pressure nonorthogonal corrections: "
            f"{self.solver.controls.pressure_nonorthogonal_max_iterations}\n"
        )

    def initial_solution(self) -> tuple[TensorLike, TensorLike, TensorLike]:
        """Return the initial velocity, face velocity, and pressure fields."""
        t0 = self.solver.controls.duration[0]
        cell_center = self.fvm_geometry.cell_center
        face_center = self.fvm_geometry.face_center
        U0 = self.pde.velocity_0(cell_center, t0)
        Uf0 = self.pde.velocity_0(face_center, t0)
        p0 = self.pde.pressure_0(cell_center, t0)
        return U0, Uf0, p0

    def solve(
        self,
        *,
        snapshot_callback: PisoSnapshotCallback | None = None,
        corrector_callback: PisoCorrectorCallback | None = None,
    ) -> PisoSolveResult:
        """Run one PISO solve from the PDE initial fields."""
        initial_velocity, initial_face_velocity, initial_pressure = (
            self.initial_solution()
        )
        return self.solver.solve(
            initial_velocity,
            initial_face_velocity,
            initial_pressure,
            snapshot_callback=snapshot_callback,
            corrector_callback=corrector_callback,
        )

    def compute_error(
        self,
        result: PisoSolveResult,
    ) -> tuple[float, ...]:
        """Compute final-time errors against exact control-volume averages."""
        t = self.solver.controls.duration[1]

        def exact_velocity(points):
            return self.pde.velocity(points, t)

        def exact_pressure(points):
            return self.pde.pressure(points, t)

        velocity_error, self.exact_velocity = cell_average_l2_error(
            self.mesh,
            exact_velocity,
            result.velocity,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        pressure_error, self.exact_pressure = cell_average_l2_error(
            self.mesh,
            exact_pressure,
            result.pressure,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        return tuple(velocity_error[i] for i in range(self.GD)) + (pressure_error,)

    def plot(self, result: PisoSolveResult) -> None:
        """Plot numerical and exact solution errors for u, v, and p."""
        import matplotlib.pyplot as plt

        t = self.solver.controls.duration[1]

        def exact_velocity(points):
            return self.pde.velocity(points, t)

        def exact_pressure(points):
            return self.pde.pressure(points, t)

        _, exact_cell_velocity = cell_average_l2_error(
            self.mesh,
            exact_velocity,
            result.velocity,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        _, exact_cell_pressure = cell_average_l2_error(
            self.mesh,
            exact_pressure,
            result.pressure,
            q=self.error_quadrature_order,
            geometry=self.fvm_geometry,
        )
        cell_centers = self.fvm_geometry.cell_center
        x, y = cell_centers[:, 0], cell_centers[:, 1]

        fig = plt.figure(figsize=(15, 10))
        titles = [
            (
                "Error u",
                result.velocity[:, 0] - exact_cell_velocity[:, 0],
            ),
            (
                "Error v",
                result.velocity[:, 1] - exact_cell_velocity[:, 1],
            ),
            (
                "Error p",
                result.pressure - exact_cell_pressure,
            ),
        ]
        for i, (title, data) in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            ax.plot_trisurf(x, y, data, cmap="viridis")
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
