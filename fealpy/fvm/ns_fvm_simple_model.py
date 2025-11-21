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
    ScalarSourceIntegrator,
    ConvectionIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
)


class NSFVMSimpleModel(ComputationalModel):
    """
    Finite Volume SIMPLE solver for 2D steady incompressible Navier–Stokes equations.
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

    def set_pde(self, pde: Union[int, object]) -> None:
        """Set PDE model (Navier–Stokes example)."""
        self.pde = PDEModelManager("navier_stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int, ny: int) -> None:
        """Initialize computational mesh."""
        self.mesh = self.pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)
        self.cm = self.mesh.entity_measure('cell')
        self.NC = self.mesh.number_of_cells()

    def set_space(self, degree: int) -> None:
        """Define pressure/velocity function spaces."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))

    # --- SIMPLE components ----------------------------------------------------

    def temporary_velocity(self, p, uf) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum eqn for intermediate velocity u*."""
        # Diffusion
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        B = bform.assembly()

        # Source
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)

        C = BilinearForm(self.velocity_space).add_integrator(
            ConvectionIntegrator(q=self.p + 2, coef=uf)).assembly()
        B = B + C
        # f = dbc.ConvectionApplyX(f)

        ap = B.diags().values

        grad_p = GradientReconstruct(self.mesh).AverageGradientreNeumann(
            p, self.pde.neumann_pressure
        )  # (NC, 2)
        p1 = bm.einsum('i,i->i', grad_p[:,0], self.cm)
        p2 = bm.einsum('i,i->i', grad_p[:,1], self.cm)
        p_grad_integrator = bm.concatenate((p1,p2))
        f = f - p_grad_integrator
        u = spsolve(B, f, "mumps")
        return ap, u

    def Ucell2edge(self, u, gd):
        """Interpolate cell-centered u to face velocities uf."""
        u = bm.stack([u[:self.NC], u[self.NC:]], axis=-1)
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter("edge")
        e2c = self.mesh.edge_to_cell()

        bdedgepoint = edge_middle_point[bd_edge]
        bdedgeu = gd(bdedgepoint)

        uf = bm.zeros((self.mesh.number_of_faces(), 2), dtype=u.dtype)
        uf += (u[e2c[:, 0]] + u[e2c[:, 1]]) / 2
        uf[bd_edge, :] = bdedgeu
        uf = bm.reshape(uf, (-1, 2))
        return uf

    def rhie_chow(self, uf, ph0, ap_edge):
        """Rhie-Chow interpolation to face velocity."""
        avargrad_p = GradientReconstruct(self.mesh).AverageGradientreNeumann(
            ph0, self.pde.neumann_pressure
        )  # (NC, 2)
        avargrad_f = GradientReconstruct(self.mesh).reconstruct(avargrad_p)
        x = self.mesh.boundary_face_index()
        e2c = self.mesh.edge_to_cell()
        dp = self.cm / ap_edge[:len(self.cm)]
        df = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        mask = bm.ones(avargrad_f.shape[0], dtype=bool)
        mask[x] = False
        uf[mask] += bm.einsum("i,ij->ij", df[mask], avargrad_f[mask])
        return uf,df


    def pressure_correct(self, ap: TensorLike, uf: TensorLike,p) -> TensorLike:
        """Solve pressure correction equation."""
        div_u = DivergenceReconstruct(self.mesh).Reconstruct(uf)  # (NC,)
        
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=2))
        A = bform2.assembly()

        LagA = self.mesh.entity_measure("cell")
        div_u = ap[:len(LagA)]*div_u
        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                 bm.arange(len(LagA), dtype=bm.int32)]),LagA,
            spshape=(1, len(LagA)),
        )
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format="csr")

        b0 = bm.array([0])
        b = bm.concatenate([-div_u, b0], axis=0)

        sol = spsolve(A, b, "mumps")
        p_c = sol[:-1]
        return p_c

    def solve(self, max_iter: int = 100, tol: float = 1e-5) -> Tuple[TensorLike, TensorLike]:
        """Main SIMPLE loop."""
        p = bm.zeros(self.NC)
        uf = bm.zeros((self.mesh.number_of_faces(), 2))

        ap, u = self.temporary_velocity(p, uf)
        self.residuals = []

        for i in range(max_iter):
            uf = self.Ucell2edge(u, self.pde.dirichlet_velocity)
            p_c = self.pressure_correct(ap, uf,p)
            L2p_c = bm.sqrt(bm.sum(self.cm * (p_c) ** 2))
            self.residuals.append(float(L2p_c))
            self.logger.info(f"[Iter {i+1}] pressure_correct = {L2p_c:.4e}")

            if L2p_c < tol:
                self.logger.info("Converged.")
                break

            p += 0.8*p_c
            _, u = self.temporary_velocity(p, uf)

        self.uh = u[:self.NC]
        self.vh = u[self.NC:]
        self.ph = p
        return self.uh, self.vh, self.ph

    def compute_error(self) -> Tuple[float, float]:
        """Compute errors for velocity and pressure."""
        cell_centers = self.mesh.entity_barycenter('cell')
        self.uI = self.pde.velocity(cell_centers)[:, 0]
        self.vI = self.pde.velocity(cell_centers)[:, 1]
        self.pI = self.pde.pressure(cell_centers)
        uerror = bm.sqrt(bm.sum(self.cm * (self.uh - self.uI)**2))
        verror = bm.sqrt(bm.sum(self.cm * (self.vh - self.vI)**2))
        perror = bm.sqrt(bm.sum(self.cm * (self.ph - self.pI)**2))
        # uerror = bm.max(bm.abs(self.uh - self.uI))
        # verror = bm.max(bm.abs(self.vh - self.vI))
        # perror = bm.max(bm.abs(self.ph - self.pI))
        return uerror, verror, perror

    def plot(self) -> None:
        """Plot numerical and exact solutions for u, v, and p."""
        import matplotlib.pyplot as plt
        cell_centers = self.mesh.entity_barycenter('cell')
        x, y = cell_centers[:, 0], cell_centers[:, 1]

        fig = plt.figure(figsize=(15, 10))
        titles = [
            ("Error u", self.uh - self.uI),
            ("Error v", self.vh - self.vI),
            ("Error p", self.ph - self.pI),
        ]
        for i, (title, data) in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            ax.plot_trisurf(x, y, data, cmap='viridis')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_residual(self) -> None:
        """Plot residual decay curve."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.residuals, marker="o", linestyle="-", color="b")
        plt.title("Pressure Correction Residual vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
