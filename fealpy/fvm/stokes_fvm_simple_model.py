from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.solver import spsolve

from . import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
)

class StokesFVMSimpleModel(ComputationalModel):
    """
    The Stokes equation in two-dimensional cases is solved by the finite volume method
    using the SIMPLE algorithm. The velocity and pressure are iteratively corrected
    to satisfy the continuity equation.
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

    def set_pde(self, pde: Union[str, object]) -> None:
        """Set the PDE model."""
        self.pde = PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        """Set the computational mesh."""
        self.mesh = self.pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)
        self.cm = self.mesh.entity_measure('cell')
        

    def set_space(self, degree: int = 0) -> None:
        """Set the function spaces for velocity and pressure."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))
        self.NC = self.mesh.number_of_cells()

    def temporary_velocity(self, p) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        B = bform.assembly()
        ap = B.diags().values
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)
        
        grad_p = GradientReconstruct(self.mesh).AverageGradientreNeumann(p, self.pde.neumann_pressure)  # (NC, 2)
        # grad_p = GradientReconstruct(self.mesh).AverageGradientreDirichlet(p, self.pde.dirichlet_pressure)  # (NC, 2)
        p1 = bm.einsum('i,i->i', grad_p[:,0], self.cm)
        p2 = bm.einsum('i,i->i', grad_p[:,1], self.cm)
        p_grad_integrator = bm.concatenate((p1,p2))
        f = f - p_grad_integrator

        u = spsolve(B, f,"mumps")
        return ap, u
    
    def Ucell2edge(self,u,gd):
        u = bm.stack([u[:self.NC],u[self.NC:]],axis=-1)
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge')
        e2c = self.mesh.edge_to_cell()
        bdedgepoint = edge_middle_point[bd_edge]
        bdedgeu = gd(bdedgepoint)
        uf = bm.zeros((self.mesh.number_of_faces(), 2), dtype=u.dtype)
        uf += (u[e2c[:,0]]+u[e2c[:,1]])/2
        uf[bd_edge, :] = bdedgeu
        uf = bm.reshape(uf, (-1, 2))
        return uf
    
    def pressure_correct(self, ap: TensorLike, uf: TensorLike) -> TensorLike:
        """Solve for pressure correction p' to enforce continuity."""

        
        div_u = DivergenceReconstruct(self.mesh).Reconstruct(uf)  # (NE,)
        
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=2))
        A = bform2.assembly()
        LagA = self.mesh.entity_measure('cell')
        div_u = ap[:len(LagA)]*div_u
        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                             bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b = bm.concatenate([-div_u, b0], axis=0)
        sol = spsolve(A, b,"mumps")
        p_c = sol[:-1]
        return p_c

    def solve(self, max_iter: int = 10, tol: float = 1e-5) -> Tuple[TensorLike, TensorLike]:
        """Solve the Stokes equation using the SIMPLE algorithm."""
        p = bm.zeros(self.NC)
        ap, u = self.temporary_velocity(p)
        self.residuals = []
        for i in range(max_iter):
            uf = self.Ucell2edge(u, self.pde.dirichlet_velocity)
            p_corr = self.pressure_correct(ap, uf)
            L2_p_corr = bm.sqrt(bm.sum(self.cm * (p_corr)**2))
            self.residuals.append(float(L2_p_corr))
            self.logger.info(f"[Iter {i+1}] L2 norm of the pressure correction : {L2_p_corr:.6e}")
            if L2_p_corr < tol:
                self.logger.info("Converged.")
                break
            p += 0.8*p_corr
            _, u = self.temporary_velocity(p)

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

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.residuals, marker='o', linestyle='-', color='b')
        plt.title("Pressure Correction Residual vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()