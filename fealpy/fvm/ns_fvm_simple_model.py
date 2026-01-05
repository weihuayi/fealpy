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
    RhieChowInterpolation
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

    def sum_duplicates_csr_manual(self,csr):
        from fealpy.sparse import csr_matrix
        indptr = csr.indptr       # shape (nrow+1,)
        indices = csr.indices     # shape (nnz,)
        data = csr.data           # shape (nnz,)
        nrow, ncol = csr.shape
        counts = indptr[1:] - indptr[:-1]
        row = bm.repeat(bm.arange(nrow), counts)
        flat_idx = row * ncol + indices
        summed = bm.bincount(flat_idx, weights=data, minlength=nrow*ncol)
        nnz_idx = bm.nonzero(summed)[0]
        new_data = summed[nnz_idx]
        new_row, new_col = divmod(nnz_idx, ncol)
        return csr_matrix((new_data, (new_row, new_col)), shape=csr.shape)

# --- SIMPLE components ----------------------------------------------------

    def temporary_velocity(self, p, uf) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum eqn for intermediate velocity u*."""
        # Diffusion
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        # Convection
        bform.add_integrator(ConvectionIntegrator(q=self.p + 2, coef=uf))
        B = bform.assembly()
        # Source
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)
        #fealpy中的稀疏矩阵是有问题的,需要手动合并重复的行列项
        B = self.sum_duplicates_csr_manual(B)
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

    def pressure_correct(self, ap: TensorLike, uf: TensorLike,p) -> TensorLike:
        """Solve pressure correction equation.
        在这里实际上是有问题的,理论上计算dp_edge的程序应该是:
        dp = cm/ap[:len(cm)]
        dp_edge = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        但是随着网格加密,使用此dp_edge会导致求出的压力修正过大,必须使用非常小的松弛因子比如0.001才能使迭代收敛.
        所以进行了改动,使用dp = 1/ap[:len(cm)],然后计算
        dp_edge = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        dp_edge = dp_edge*em
        最终不需要过大的收敛因子,而且迭代次数也相对会减少一些.
        但这样做,在算法原理上是矛盾的,而且随着网格的加密,松弛因子仍然要不断减小,大概是网格加密一倍,松弛因子就要减小到原来的一半.
        这个问题需要进一步研究.
        """
        cm = self.mesh.entity_measure('cell')
        em = self.mesh.entity_measure('edge')
        dp = 1/ap[:len(cm)]
        e2c = self.mesh.edge_to_cell()
        dp_edge = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        dp_edge = dp_edge*em
        div_u = DivergenceReconstruct(self.mesh).Reconstruct(uf)  # (NC,)
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=2,coef=dp_edge))
        A = bform2.assembly()
        LagA = self.mesh.entity_measure("cell")
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
        # dp = self.cm/ap[:self.NC]
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge')
        bdedgepoint = edge_middle_point[bd_edge]
        bdedgeu = self.pde.dirichlet_velocity(bdedgepoint)
        for i in range(max_iter):
            uf = RhieChowInterpolation(self.mesh).Interpolation(u,ap,p)
            uf[bd_edge, :] = bdedgeu
            p_c = self.pressure_correct(ap, uf,p)
            L2p_c = bm.sqrt(bm.sum(self.cm * (p_c) ** 2))
            self.residuals.append(float(L2p_c))
            self.logger.info(f"[Iter {i+1}] pressure_correct = {L2p_c}")

            if L2p_c < tol:
                self.logger.info("Converged.")
                break

            p += 0.4*p_c

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