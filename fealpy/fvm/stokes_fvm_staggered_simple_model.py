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
    StaggeredMeshManager,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
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
        self.div = DivergenceReconstruct(self.pmesh)
        self.pcm = self.pmesh.entity_measure("cell")
        self.ucm = self.umesh.entity_measure("cell")
        self.vcm = self.vmesh.entity_measure("cell")
        self.ppoints = self.pmesh.entity_barycenter("cell")
        self.upoints = self.umesh.entity_barycenter("cell")
        self.vpoints = self.vmesh.entity_barycenter("cell")

    def compute_velocity_u(self, p_u) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        uspace = ScaledMonomialSpace2d(self.umesh, 0)
        A = BilinearForm(uspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(uspace).add_integrator(ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        grad_p = GradientReconstruct(self.umesh).AverageGradientreNeumann(p_u, self.pde.neumann_pressure)
        # grad_p = GradientReconstruct(self.umesh).AverageGradientreDirichlet(p_u, self.pde.dirichlet_pressure)
        f -= bm.einsum('i,i->i', grad_p[:, 0], self.ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        uap = A.diags().values
        return spsolve(A, f,"mumps"), uap

    def compute_velocity_v(self, p_v) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)
        A = BilinearForm(vspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(vspace).add_integrator(ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        grad_p = GradientReconstruct(self.vmesh).AverageGradientreNeumann(p_v,self.pde.neumann_pressure)
        # grad_p = GradientReconstruct(self.vmesh).AverageGradientreDirichlet(p_v,self.pde.dirichlet_pressure)
        f -= bm.einsum('i,i->i', grad_p[:, 1], self.vcm)
        dbc = DirichletBC(self.vmesh, self.pde.dirichlet_velocity_v,
                          threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        vap = A.diags().values
        return spsolve(A, f,"mumps"), vap

    def correct_pressure_compute(self, f: TensorLike, a_p_edge: TensorLike) -> TensorLike:
        """
        Solve for pressure correction p' to enforce continuity.
        """
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2,coef=1 / a_p_edge)
        ).assembly()  
        LagA = self.pmesh.entity_measure('cell')
        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                             bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b = bm.concatenate([f, b0], axis=0)
        sol = spsolve(A, b)
        p_correct = sol[:-1]   
        return p_correct

    def solve(self, max_iter: int = 200, tol: float = 1e-6) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """
        Solve the Stokes equation using the SIMPLE algorithm.
        """
        p = bm.zeros(self.ppoints.shape[0])
        self.residuals = []
        for i in range(max_iter):
            
            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p)
            uh, a_p_u = self.compute_velocity_u(p_u)
            vh, a_p_v = self.compute_velocity_v(p_v)
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
            self.div_rhs = self.div.StagReconstruct(edge_vel)
            p_corr = self.correct_pressure_compute(-self.div_rhs, a_p_edge)
            L2_p_corr = bm.sqrt(bm.sum(self.pcm * (p_corr)**2))
            self.residuals.append(float(L2_p_corr))
            self.logger.info(f"[Iter {i+1}] L2 norm of the pressure correction : {L2_p_corr:.6e}")
            if L2_p_corr < tol:
                self.logger.info("Converged.")
                break
            p += 0.8*p_corr
        self.uh, self.vh, self.ph = uh, vh, p
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute errors for velocity and pressure.
        """
        self.uI = self.pde.velocity_u(self.upoints)
        self.vI = self.pde.velocity_v(self.vpoints)
        self.pI = self.pde.pressure(self.ppoints)
        
        uerror = bm.sqrt(bm.sum(self.ucm * (self.uh - self.uI)**2))
        verror = bm.sqrt(bm.sum(self.vcm * (self.vh - self.vI)**2))
        perror = bm.sqrt(bm.sum(self.pcm * (self.ph - self.pI)**2))
        # uerror = bm.max(bm.abs(self.uh - self.uI))
        # verror = bm.max(bm.abs(self.vh - self.vI))
        # perror = bm.max(bm.abs(self.ph - self.pI))
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
        '''Plot the residual descent curve of pressure p correction.'''
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.residuals, marker='o', linestyle='-', color='b')
        plt.title("Pressure Correction Residual vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
