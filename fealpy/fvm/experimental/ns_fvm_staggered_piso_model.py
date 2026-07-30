from typing import Tuple, Union

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor, CSRTensor
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
from .staggered_divergence_reconstruct import StaggeredDivergenceReconstruct
from fealpy.decorator import cartesian


class NSFVMStaggeredPISOModel(ComputationalModel):
    """Staggered finite-volume PISO solver for transient Navier-Stokes cases."""

    def __init__(self, options):
        self.options = options
        super().__init__(
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.duration = tuple(options.get("duration", (0, 1)))
        self.nt = options.get("nt", 40)
        self.tau = (self.duration[1] - self.duration[0]) / self.nt
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.pmesh.number_of_cells()} pressure cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
            f"  Time steps: {self.nt}\n"
        )

    def set_pde(self, pde: Union[int, object]) -> None:
        self.pde = PDEModelManager("navier_stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int, ny: int) -> None:
        self.staggered_mesh = StaggeredMeshManager(self.pde.domain(), nx=nx, ny=ny)
        self.umesh = self.staggered_mesh.umesh
        self.vmesh = self.staggered_mesh.vmesh
        self.pmesh = self.staggered_mesh.pmesh

        self.uspace = ScaledMonomialSpace2d(self.umesh, p=0)
        self.vspace = ScaledMonomialSpace2d(self.vmesh, p=0)
        self.pspace = ScaledMonomialSpace2d(self.pmesh, p=0)

        self.div = StaggeredDivergenceReconstruct(self.pmesh)
        self.pcm = self.pmesh.entity_measure("cell")
        self.ucm = self.umesh.entity_measure("cell")
        self.vcm = self.vmesh.entity_measure("cell")
        self.ppoints = self.pmesh.entity_barycenter("cell")
        self.upoints = self.umesh.entity_barycenter("cell")
        self.vpoints = self.vmesh.entity_barycenter("cell")

        self.UNC = self.umesh.number_of_cells()
        self.VNC = self.vmesh.number_of_cells()
        self.PNC = self.pmesh.number_of_cells()

        self.ue2c = self.umesh.edge_to_cell()
        self.ve2c = self.vmesh.edge_to_cell()
        self.ucell2pedge = self.staggered_mesh.get_dof_mapping_ucell2pedge()
        self.vcell2pedge = self.staggered_mesh.get_dof_mapping_vcell2pedge()
        self.vedge2uedge = self.staggered_mesh.get_dof_mapping_vedge2uedge()
        self.uedge2vedge = self.staggered_mesh.get_dof_mapping_uedge2vedge()
        self.u_gradient = GradientReconstruct(self.umesh)
        self.v_gradient = GradientReconstruct(self.vmesh)

    def initial_solution(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        t0 = self.duration[0]
        u0 = self.pde.velocity_u0(self.upoints, t0)
        v0 = self.pde.velocity_v0(self.vpoints, t0)
        p0 = self.pde.pressure_0(self.ppoints, t0)
        return u0, v0, p0

    def compute_temporary_velocity_u(
        self,
        u0: TensorLike,
        v0: TensorLike,
        p_u: TensorLike,
        t: float,
    ) -> Tuple[TensorLike, TensorLike]:
        """Solve the u-momentum predictor in the scaled backward Euler form."""
        uf0 = (u0[self.ue2c[:, 0]] + u0[self.ue2c[:, 1]]) / 2
        vf0 = (v0[self.ve2c[:, 0]] + v0[self.ve2c[:, 1]]) / 2
        vf0_umesh = vf0[self.vedge2uedge.astype(int)]
        face_velocity = bm.stack([uf0, vf0_umesh], axis=1)

        bform = BilinearForm(self.uspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2, coef=face_velocity))
        A = bform.assembly()

        mass_matrix = CSRTensor(
            crow=bm.arange(self.UNC + 1),
            col=bm.arange(self.UNC),
            values=self.ucm,
            spshape=(self.UNC, self.UNC),
        )

        @cartesian
        def source(points):
            return self.pde.source_u(points, t)

        rhs = LinearForm(self.uspace).add_integrator(
            ScalarSourceIntegrator(source, q=2)
        ).assembly()

        grad_p = self.u_gradient.cell_gradient(p_u)
        pressure_term = bm.einsum("i,i->i", grad_p[:, 0], self.ucm)
        old_mass = bm.einsum("i,i->i", u0, self.ucm)

        A = A * self.tau + mass_matrix
        rhs = self.tau * (rhs - pressure_term) + old_mass

        dbc = DirichletBC(
            self.umesh,
            self.pde.velocity_dirichlet_u,
            threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10),
        )
        A, rhs = dbc.apply_diffusion(A, rhs, components=1)
        A, rhs = dbc.ThresholdApply(A, rhs)
        a_p = A.diags().values
        return spsolve(A, rhs, "mumps"), a_p

    def compute_temporary_velocity_v(
        self,
        u0: TensorLike,
        v0: TensorLike,
        p_v: TensorLike,
        t: float,
    ) -> Tuple[TensorLike, TensorLike]:
        """Solve the v-momentum predictor in the scaled backward Euler form."""
        uf0 = (u0[self.ue2c[:, 0]] + u0[self.ue2c[:, 1]]) / 2
        vf0 = (v0[self.ve2c[:, 0]] + v0[self.ve2c[:, 1]]) / 2
        uf0_vmesh = uf0[self.uedge2vedge.astype(int)]
        face_velocity = bm.stack([uf0_vmesh, vf0], axis=1)

        bform = BilinearForm(self.vspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2, coef=face_velocity))
        A = bform.assembly()

        mass_matrix = CSRTensor(
            crow=bm.arange(self.VNC + 1),
            col=bm.arange(self.VNC),
            values=self.vcm,
            spshape=(self.VNC, self.VNC),
        )

        @cartesian
        def source(points):
            return self.pde.source_v(points, t)

        rhs = LinearForm(self.vspace).add_integrator(
            ScalarSourceIntegrator(source, q=2)
        ).assembly()

        grad_p = self.v_gradient.cell_gradient(p_v)
        pressure_term = bm.einsum("i,i->i", grad_p[:, 1], self.vcm)
        old_mass = bm.einsum("i,i->i", v0, self.vcm)

        A = self.tau * A + mass_matrix
        rhs = self.tau * (rhs - pressure_term) + old_mass

        dbc = DirichletBC(
            self.vmesh,
            self.pde.velocity_dirichlet_v,
            threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10),
        )
        A, rhs = dbc.apply_diffusion(A, rhs, components=1)
        A, rhs = dbc.ThresholdApply(A, rhs)
        a_p = A.diags().values
        return spsolve(A, rhs, "mumps"), a_p

    def pressure_response_on_pedge(self, a_p_edge: TensorLike) -> TensorLike:
        """Map velocity-control-volume response ``V/a_p`` to pressure faces."""
        response = bm.zeros_like(a_p_edge)
        response[self.ucell2pedge] = self.ucm / a_p_edge[self.ucell2pedge]
        response[self.vcell2pedge] = self.vcm / a_p_edge[self.vcell2pedge]
        return response

    def correct_pressure_compute(self, rhs: TensorLike, a_p_edge: TensorLike) -> TensorLike:
        """Solve the pressure correction equation with a zero-mean constraint."""
        response = self.pressure_response_on_pedge(a_p_edge)
        A = BilinearForm(self.pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2, coef=response)
        ).assembly()

        constraint = COOTensor(
            bm.array([
                bm.zeros(self.PNC, dtype=bm.int32),
                bm.arange(self.PNC, dtype=bm.int32),
            ]),
            self.pcm,
            spshape=(1, self.PNC),
        )
        A = BlockForm([[A, constraint.T], [constraint, None]])
        A = A.assembly_sparse_matrix(format="csr")
        b = bm.concatenate([rhs, bm.array([0])], axis=0)
        sol = spsolve(A, b, "scipy")
        return sol[:-1]

    def velocity_pressure_correction(
        self,
        u: TensorLike,
        v: TensorLike,
        p_corr: TensorLike,
        uap: TensorLike,
        vap: TensorLike,
    ) -> Tuple[TensorLike, TensorLike]:
        """Apply ``U <- U - (V/a_p) grad(p')`` on staggered velocity cells."""
        u_pcorr, v_pcorr = self.staggered_mesh.map_pressure_pcell_to_uvedge(
            p_corr
        )
        ugrad_p = self.u_gradient.cell_gradient(u_pcorr)
        vgrad_p = self.v_gradient.cell_gradient(v_pcorr)
        u_new = u - self.ucm / uap * ugrad_p[:, 0]
        v_new = v - self.vcm / vap * vgrad_p[:, 1]
        return u_new, v_new

    def solve(
        self,
        u0: TensorLike = None,
        v0: TensorLike = None,
        p0: TensorLike = None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        if u0 is None or v0 is None or p0 is None:
            u0, v0, p0 = self.initial_solution()

        u3, v3, p2 = u0, v0, p0
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p0)

            u1, uap = self.compute_temporary_velocity_u(u0, v0, p_u, t + self.tau)
            v1, vap = self.compute_temporary_velocity_v(u0, v0, p_v, t + self.tau)

            edge_vel1, a_p_edge = self.staggered_mesh.map_velocity_uvcell_to_pedge(
                u1, v1, uap, vap
            )
            div_rhs1 = self.div.StagReconstruct(edge_vel1)
            p_corr1 = self.correct_pressure_compute(-div_rhs1, a_p_edge)
            p1 = p0 + p_corr1
            u2, v2 = self.velocity_pressure_correction(
                u1, v1, p_corr1, uap, vap
            )

            edge_vel2, _ = self.staggered_mesh.map_velocity_uvcell_to_pedge(
                u2, v2, uap, vap
            )
            div_rhs2 = self.div.StagReconstruct(edge_vel2)
            p_corr2 = self.correct_pressure_compute(-div_rhs2, a_p_edge)
            p2 = p1 + p_corr2
            u3, v3 = self.velocity_pressure_correction(
                u2, v2, p_corr2, uap, vap
            )

            u0, v0, p0 = u3, v3, p2

        self.uh = u3
        self.vh = v3
        self.ph = p2
        self.edge_vel, _ = self.staggered_mesh.map_velocity_uvcell_to_pedge(
            self.uh, self.vh, uap, vap
        )
        return self.uh, self.vh, self.ph

    def compute_error(self) -> Tuple[float, float, float]:
        t = self.duration[1]
        q = getattr(self, "error_quadrature_order", 4)

        def exact_u(points):
            return self.pde.velocity_u(points, t)

        def exact_v(points):
            return self.pde.velocity_v(points, t)

        def exact_p(points):
            return self.pde.pressure(points, t)

        uerror, self.uI = cell_average_l2_error(self.umesh, exact_u, self.uh, q=q)
        verror, self.vI = cell_average_l2_error(self.vmesh, exact_v, self.vh, q=q)
        perror, self.pI = cell_average_l2_error(self.pmesh, exact_p, self.ph, q=q)
        return uerror, verror, perror

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        if not hasattr(self, "uI"):
            self.compute_error()

        fig = plt.figure(figsize=(15, 5))

        px, py = self.ppoints.T
        ux, uy = self.upoints.T
        vx, vy = self.vpoints.T

        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.plot_trisurf(ux, uy, self.uh - self.uI, cmap="viridis")
        ax1.set_title("Error u")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.plot_trisurf(vx, vy, self.vh - self.vI, cmap="viridis")
        ax2.set_title("Error v")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot_trisurf(px, py, self.ph - self.pI, cmap="viridis")
        ax3.set_title("Error p")

        plt.tight_layout()
        plt.show()
