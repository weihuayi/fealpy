from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarCrossDiffusionIntegrator,
    ScalarSourceIntegrator,
    FVMGeometry,
    GradientReconstruct,
    reconstruct_face_gradient,
    ConvectionIntegrator,
    cell_average_l2_error,
)
from .legacy_boundary_conditions import (
    ExperimentalDirichletBC as DirichletBC,
    ExperimentalNeumannBC as NeumannBC,
)
from .rhie_chow_coupled_operator import RhieChowCoupledOperator


class StokesFVMRCModel(ComputationalModel):
    """
    A 2D Stokes equation solver using the finite volume method (FVM).

    This computational model solves the 2D Stokes equation on a uniform grid,
    incorporating Rhie-Chow interpolation correction to mitigate oscillations in the numerical pressure solution.

    Parameters:
        options (dict): Configuration dictionary for model setup.
            - 'pde': PDE data or index
            - 'nx', 'ny': mesh divisions
            - 'pbar_log', 'log_level': logging controls

    Attributes:
        mesh : The initialized computational mesh.
        pde : The PDE model object.
        uh,vh,ph : Numerical solution vector (computed after calling `solve_rhie_chow`).

    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "INFO"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space()


    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh: {self.mesh.number_of_cells()} cells\n"
            f"  Space degree: {self.p}\n"
            f"  PDE: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[str, object]) -> None:
        if isinstance(pde, int):
            self.pde = PDEModelManager('stokes').get_example(pde)
        else:
            self.pde = pde

        self.logger.info(self.pde)

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        self.mesh_type = self.options.get("mesh_type", "uniform_qrad")
        self.mesh = self.pde.init_mesh[self.mesh_type](nx=nx, ny=ny)
        self.NC = self.mesh.number_of_cells()

    def set_space(self, degree: int = 0) -> None:
        self.p = degree
        self.pspace = ScaledMonomialSpace2d(self.mesh, self.p)
        self.uspace = TensorFunctionSpace(self.pspace, shape=(2, -1))
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            method="green_gauss",
            boundary_value=self.pde.dirichlet_velocity,
            boundary_type="dirichlet",
        )
        self.fvm_geometry = FVMGeometry(self.mesh)
        self.velocity_dirichlet_bc = DirichletBC(
            self.mesh, self.pde.dirichlet_velocity
        )
        self.rc_operator = RhieChowCoupledOperator(self.mesh)

    def assembly_velocity(self, u0=None) -> Tuple[TensorLike, TensorLike]:
        """
        Discretize the velocity term
        """
        AB = BilinearForm(self.uspace).add_integrator(
            ScalarDiffusionIntegrator(q=2)).assembly()

        f = LinearForm(self.uspace).add_integrator(
            ScalarSourceIntegrator(self.pde.source, q=2)).assembly()
        if u0 is not None:
            f = f + self.compute_cross_diffusion(u0)

        return AB, f

    def compute_cross_diffusion(self, uh: TensorLike) -> TensorLike:
        """Assemble the explicit non-orthogonal diffusion correction."""
        lform = LinearForm(self.uspace)
        U = bm.stack((uh[:self.NC], uh[self.NC:]), axis=1)
        grad_u = self.velocity_gradient.cell_gradient(U)
        grad_f = reconstruct_face_gradient(self.mesh, grad_u)
        lform.add_integrator(
            ScalarCrossDiffusionIntegrator(
                uh,
                grad_f,
                geometry=self.fvm_geometry,
                method="bounded_over_relaxed",
                boundary_policy="all",
            )
        )
        return lform.assembly()

    def assembly_pressure(self) -> Tuple[TensorLike, TensorLike]:
        """
        Discretize the pressure term
        """
        NE = self.mesh.number_of_edges()
        c = bm.tile([1, 0], (NE, 1))
        d = bm.tile([0, 1], (NE, 1))

        M1 = BilinearForm(self.pspace).add_integrator(
            ConvectionIntegrator(q=2, coef=c)).assembly()
        M2 = BilinearForm(self.pspace).add_integrator(
            ConvectionIntegrator(q=2, coef=d)).assembly()

        return M1, M2

    def lagrange_multiplier(self) -> TensorLike:
        """
        Ensure the uniqueness of the numerical pressure solution
        under pure Neumann boundary conditions using the Lagrange multiplier method
        """
        LagA = self.mesh.entity_measure('cell')
        A1 = COOTensor(
            bm.stack([
                bm.zeros(len(LagA), dtype=bm.int32),
                bm.arange(2 * len(LagA), 3 * len(LagA), dtype=bm.int32)
            ], axis=0),
            LagA, spshape=(1, 3 * len(LagA))
        )
        return A1

    def assembly_base_system(self, u0=None) -> Tuple:
        """
        Apply boundary conditions to the discretized velocity
        and pressure terms, and assemble them into basic matrix blocks using BlockForm
        """
        AB, f = self.assembly_velocity(u0)
        M1, M2 = self.assembly_pressure()
        M3 = BlockForm([[M1, M2]]).assembly_sparse_matrix(format='csr')
        # nbc = NeumannBC(self.mesh, self.pde.neumann_pressure)
        nbc = NeumannBC(self.mesh)
        AB, f = self.velocity_dirichlet_bc.apply_diffusion(
            AB,
            f,
            components=self.GD,
        )
        ap = self._matrix_diagonal(AB)

        M1 = nbc.ConvectionApplyX(M1, f[:self.NC])
        M2 = nbc.ConvectionApplyY(M2, f[self.NC:])

        M4 = BlockForm([[M1], [M2]]).assembly_sparse_matrix(format='csr')

        return AB, M3, M4, f, ap

    def _matrix_diagonal(self, A):
        diag_entries = A.diags()
        diag = bm.zeros(A.shape[0], dtype=diag_entries.values.dtype)
        diag = bm.index_add(diag, diag_entries.indices, diag_entries.values, axis=0)
        return diag

    def _rc_boundary_rhs(self, rc_operator):
        bd_face = self.mesh.boundary_face_index()
        bd_point = self.mesh.entity_barycenter("edge")[bd_face]
        bd_velocity = self.pde.dirichlet_velocity(bd_point)
        return rc_operator.boundary_velocity_rhs(bd_velocity)


    def solve_rhie_chow(self, return_diagnostics: bool = False) -> Tuple:
        """
        Solve the collocated Stokes block system with the shared Rhie-Chow
        pressure operator.

        The old implementation first solved an unstabilized pressure field and
        then assembled a hand-written correction block from that field.  The
        current path instead inserts the compact Rhie-Chow pressure block
        directly into the coupled continuity equation:

            [ A   G  ][U] = [f]
            [ D  LRC ][p]   [b_rc]

        The wide-stencil pressure-gradient part of the Rhie-Chow correction is
        linear in pressure for Stokes, so it is assembled into the pressure
        block.  On non-orthogonal meshes the diffusion cross term is handled
        by an explicit Picard correction, while the default quadrilateral path
        remains a single coupled solve.
        """
        u0 = None
        if self.options.get("nonorthogonal_correction", self.mesh_type != "uniform_qrad"):
            u0 = bm.zeros(2 * self.NC)

        max_iter = self.options.get("nonorthogonal_max_iter", 10 if u0 is not None else 1)
        tol = self.options.get("nonorthogonal_tol", 1.0e-7)
        diagnostics = []
        for i in range(max_iter):
            AB, M3, M4, f, ap = self.assembly_base_system(u0)
            A1 = self.lagrange_multiplier()
            b0 = bm.array([self.pde.pressure_integral_target()])
            rc_operator = self.rc_operator
            LRC = rc_operator.pressure_stabilization_matrix(ap)
            explicit_pressure = rc_operator.explicit_pressure_matrix(ap)
            pressure_block = LRC - explicit_pressure
            system_block = BlockForm(
                [[AB, M4], [M3, pressure_block]]
            ).assembly_sparse_matrix(format="csr")
            system = BlockForm([[system_block, A1.T], [A1, None]])
            system = system.assembly_sparse_matrix(format="csr")

            rhs = bm.concatenate([f, self._rc_boundary_rhs(rc_operator), b0], axis=0)
            sol = spsolve(system, rhs, "mumps")
            uh = sol[:self.NC]
            vh = sol[self.NC:2 * self.NC]
            ph = sol[2 * self.NC:-1]
            u_new = bm.concatenate([uh, vh], axis=0)
            if u0 is None:
                break
            err = bm.max(bm.abs(u_new - u0))
            diagnostics.append({"iteration": i + 1, "velocity_update": float(err)})
            u0 = u_new
            if err < tol:
                break
        self.uh, self.vh, self.ph = uh, vh, ph
        self.rhie_chow_diagnostics = diagnostics
        if return_diagnostics:
            return self.uh, self.vh, self.ph, self.rhie_chow_diagnostics
        return self.uh, self.vh, self.ph


    def compute_error(self) -> Tuple:
        """
        Compute the error between the numerical solutions for velocity u, v,
        and pressure p and their analytical solutions
        """
        q = getattr(self, "error_quadrature_order", 4)
        uerror, self.uI = cell_average_l2_error(self.mesh, self.pde.velocity_u, self.uh, q=q)
        verror, self.vI = cell_average_l2_error(self.mesh, self.pde.velocity_v, self.vh, q=q)
        perror, self.pI = cell_average_l2_error(self.mesh, self.pde.pressure, self.ph, q=q)
        return uerror, verror, perror


    def plot(self) -> None:
        """
        Plot the error of the numerical solutions for velocity u, v, and pressure p
        """
        import matplotlib.pyplot as plt
        ppoints = self.mesh.entity_barycenter('cell')
        x, y = ppoints[:, 0], ppoints[:, 1]
        fig = plt.figure(figsize=(12, 8))
        for i, (data, title) in enumerate([
                (self.uh - self.uI, "Error u"),
                (self.vh - self.vI, "Error v"),
                (self.ph - self.pI, " Error p(RC)"),
                ]):
                ax = fig.add_subplot(2, 3, i+1, projection='3d')
                ax.plot_trisurf(x, y, data, cmap='viridis')
                ax.set_title(title)
        plt.tight_layout()
        plt.show()
