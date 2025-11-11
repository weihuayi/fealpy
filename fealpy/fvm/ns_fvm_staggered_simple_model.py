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
    StaggeredMeshManager,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
    NeumannBC,
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
        self.div = DivergenceReconstruct(self.pmesh)
        self.pcm = self.pmesh.entity_measure("cell")
        self.ucm = self.umesh.entity_measure("cell")
        self.vcm = self.vmesh.entity_measure("cell")
        self.ppoints = self.pmesh.entity_barycenter("cell")
        self.upoints = self.umesh.entity_barycenter("cell")
        self.vpoints = self.vmesh.entity_barycenter("cell")

    def compute_temporary_velocity_u(self, p_u, uf) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        uspace = ScaledMonomialSpace2d(self.umesh, 0)

        bform = BilinearForm(uspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2, coef=uf))
        A = bform.assembly()

        f = LinearForm(uspace).add_integrator(ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        grad_p = GradientReconstruct(self.umesh).AverageGradientreNeumann(p_u, self.pde.neumann_pressure)
        f -= bm.einsum('i,i->i', grad_p[:, 0], self.ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        uap = A.diags().values
        # print(A.to_dense())
        return spsolve(A, f,"mumps"), uap

    def compute_temporary_velocity_v(self, p_v, uf) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)

        bform = BilinearForm(vspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2, coef=uf))
        A = bform.assembly()

        f = LinearForm(vspace).add_integrator(ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        grad_p = GradientReconstruct(self.vmesh).AverageGradientreNeumann(p_v,self.pde.neumann_pressure)
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
        # print(1 / a_p_edge)
        # print(A.to_dense())
        nbc = NeumannBC(self.pmesh, self.pde.neumann_pressure)
        f = nbc.DiffusionApply(f)
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
        # p = self.pde.pressure(self.ppoints)
        UNE = self.umesh.number_of_edges()
        uf = bm.ones(UNE)
        vf = bm.ones(UNE)
        self.residuals = []

        uedge_centers = self.umesh.entity_barycenter('edge')
        vedge_centers = self.vmesh.entity_barycenter('edge')
        dist = bm.sum((uedge_centers[:, None, :] - vedge_centers[None, :, :])**2, axis=2)
        vedge2uedge = bm.argmin(dist, axis=1)
        vf_umesh = vf[vedge2uedge]
        Uf = bm.stack([uf, vf_umesh], axis=1)

        uedge_centers = self.umesh.entity_barycenter('edge')
        vedge_centers = self.vmesh.entity_barycenter('edge')
        dist = bm.sum((uedge_centers[:, None, :] - vedge_centers[None, :, :])**2, axis=2)
        uedge2vedge = bm.argmin(dist, axis=0)
        uf_vmesh = uf[uedge2vedge]
        Vf = bm.stack([uf_vmesh, vf], axis=1)
        ue2c = self.umesh.edge_to_cell()
        ve2c = self.vmesh.edge_to_cell()
        for i in range(max_iter):

            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p)
            uh, a_p_u = self.compute_temporary_velocity_u(p_u,Uf)
            vh, a_p_v = self.compute_temporary_velocity_v(p_v,Vf)
            uf1 = (uh[ue2c[:,0]] + uh[ue2c[:,1]])/2
            vf1 = (vh[ve2c[:,0]] + vh[ve2c[:,1]])/2
            vf_umesh = vf1[vedge2uedge]
            Uf = bm.stack([uf1, vf_umesh], axis=1)
            uf_vmesh = uf1[uedge2vedge]
            Vf = bm.stack([uf_vmesh, vf1], axis=1)
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
            self.div_rhs = self.div.StagReconstruct(edge_vel)
            p_corr = self.correct_pressure_compute(-self.div_rhs, a_p_edge)
            err = bm.sqrt(bm.sum(self.pcm * p_corr ** 2))
            self.residuals.append(float(err))
            self.logger.info(f"[Iter {i+1}] Pressure correction residual: {err:.2e}")
            if err < tol:
                self.logger.info("Converged.")
                break
            p += 0.8*p_corr  
        
        self.uh, self.vh, self.ph = uh, vh, p
        self.p_correct = p_corr
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute L2 errors for velocity and pressure.
        """
        self.uI = self.pde.velocity_u(self.umesh.entity_barycenter("cell"))
        self.vI = self.pde.velocity_v(self.vmesh.entity_barycenter("cell"))
        self.pI = self.pde.pressure(self.pmesh.entity_barycenter("cell"))

        uerror = bm.sqrt(bm.sum(self.umesh.entity_measure("cell") * (self.uh - self.uI)**2))
        verror = bm.sqrt(bm.sum(self.vmesh.entity_measure("cell") * (self.vh - self.vI)**2))
        perror = bm.sqrt(bm.sum(self.pmesh.entity_measure("cell") * (self.ph - self.pI)**2))
        # uerr = bm.max(bm.abs(self.uh - self.uI))
        # verr = bm.max(bm.abs(self.vh - self.vI))
        # perr = bm.max(bm.abs(self.ph - self.pI))
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
