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
    NeumannBC,
)

class StokesFVMStaggeredModel(ComputationalModel):
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
        self.staggered_mesh = StaggeredMeshManager(self.pde, nx, ny)
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
        f -= bm.einsum('i,i->i', grad_p[:, 0], self.ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        # A, f = dbc.ThresholdApply(A, f)
        A, f = dbc.DiffusionApply(A, f)
        A = sum_duplicates_csr_manual(A)
        # uap = A.diags().values
        A, f = dbc.ThresholdApply(A, f)
        uap = A.diags().values
        return spsolve(A, f,"mumps"), uap

    def compute_velocity_v(self, p_v) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)
        A = BilinearForm(vspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(vspace).add_integrator(ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        grad_p = GradientReconstruct(self.vmesh).AverageGradientreNeumann(p_v,self.pde.neumann_pressure)
        f -= bm.einsum('i,i->i', grad_p[:, 1], self.vcm)
        dbc = DirichletBC(self.vmesh, self.pde.dirichlet_velocity_v,
                          threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
        # A, f = dbc.ThresholdApply(A, f)
        A, f = dbc.DiffusionApply(A, f)
        # print(A.shape, f.shape)
        A = sum_duplicates_csr_manual(A)
        # vap = A.diags().values
        A, f = dbc.ThresholdApply(A, f)
        # print(A.shape, f.shape)
        # print(A.to_dense())
        vap = A.diags().values
        # print("vap",vap)
        return spsolve(A, f,"mumps"), vap

    def initial_pressure_compute(self, f: TensorLike,) -> TensorLike:
        """
        Solve for pressure correction p' to enforce continuity.
        """
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        A = BilinearForm(pspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = self.neumann_bc(f,self.pde.neumann_pressure)
        LagA = self.pmesh.entity_measure('cell')
        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                             bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([self.pde.pressure_integral_target()])
        # b0 = bm.array([0])
        b = bm.concatenate([f, b0], axis=0)
        sol = spsolve(A, b,"mumps")
        p_initial = sol[:-1] 
        # p_initial -=bm.sum(p_initial*self.pmesh.entity_measure('cell'))
        # p_initial += self.pde.pressure_integral_target()
        # self.pI = self.pde.pressure(self.ppoints)
        # err = bm.sqrt(bm.sum(self.cm *(self.pI-p_initial) ** 2)) 
        # print(f"Initial pressure L2 error: {err:.2e}") 
        import matplotlib.pyplot as plt
        x, y = self.ppoints[:, 0], self.ppoints[:, 1]
        self.pI = self.pde.pressure(self.ppoints)
        fig = plt.figure(figsize=(12, 8))
        for i, (data, title) in enumerate([ 
            (p_initial, "Numerical p"),
            (self.pI, "Exact p"),
            (self.pI - p_initial, "Pressure Correction p'"),
        ]):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            ax.plot_trisurf(x, y, data, cmap='viridis')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()  
        return p_initial

    def correct_pressure_compute(self, f: TensorLike, a_p_edge: TensorLike) -> TensorLike:
        """
        Solve for pressure correction p' to enforce continuity.
        """
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2, coef=1 / a_p_edge)
        ).assembly()  
        # from fealpy.sparse import spdiags
        # bd_edge = self.pmesh.boundary_face_index()
        # e2c = self.pmesh.edge_to_cell()
        # bd_integrator = bm.ones(bd_edge.shape[0])
        # bde2c = e2c[bd_edge, 0]
        # bdIdx = bdIdx = bm.zeros(A.shape[0])
        # bm.add_at(bdIdx, bde2c, bd_integrator)
        # B = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        # A = A - B
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
        self.residuals = []
        for i in range(max_iter):
            p_u, p_v = self.staggered_mesh.interpolate_pressure_to_edges(p)
            uh, a_p_u = self.compute_velocity_u(p_u)
            vh, a_p_v = self.compute_velocity_v(p_v)
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_cell_to_edge(uh, vh, a_p_u, a_p_v)
            self.div_rhs = self.div.StagReconstruct(edge_vel)
            p_corr = self.correct_pressure_compute(-self.div_rhs, a_p_edge)
            err = bm.sqrt(bm.sum(self.pcm * p_corr ** 2))
            self.residuals.append(float(err))
            self.logger.info(f"[Iter {i+1}] Pressure correction residual: {err:.2e}")
            if err < tol:
                self.logger.info("Converged.")
                break
            p += p_corr  
        
        self.pI = self.pde.pressure(self.ppoints)
        self.pgrad_ph = GradientReconstruct(self.pmesh).AverageGradientreNeumann(p,self.pde.neumann_pressure)
        self.pgrad_pI1 = GradientReconstruct(self.pmesh).AverageGradientreNeumann(self.pI,self.pde.neumann_pressure)
        self.pgrad_pI2 = self.pde.grad_pressure(self.ppoints)
        # print(bm.sqrt(bm.sum(self.pcm * (self.pgrad_ph[:,0] - self.pgrad_pI1[:,0])**2)))
        # print(bm.sqrt(bm.sum(self.pcm * (self.pgrad_ph[:,0] - self.pgrad_pI2[:,0])**2)))
        self.uh, self.vh, self.ph = uh, vh, p
        self.p_correct = p_corr
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute L2 errors for velocity and pressure.
        """
        self.uI = self.pde.velocity_u(self.upoints)
        self.vI = self.pde.velocity_v(self.vpoints)
        self.pI = self.pde.pressure(self.ppoints)
        ue = bm.sqrt(bm.sum(self.ucm * (self.uh - self.uI)**2))
        ve = bm.sqrt(bm.sum(self.vcm * (self.vh - self.vI)**2))
        pe0 = bm.sqrt(bm.sum(self.pcm * (self.ph - self.pI)**2))
        # pe = self.ph - self.pI
        return ue, ve, pe0

    def plot(self) -> None:
        """Plot numerical and exact solutions for u1, u2, and p."""
        import matplotlib.pyplot as plt
        x, y = self.ppoints[:, 0], self.ppoints[:, 1]

        fig = plt.figure(figsize=(12, 8))
        for i, (data, title) in enumerate([
            (self.pI-self.ph, "Numerical p"),
            (self.pgrad_ph[:,0] - self.pgrad_pI1[:,0], "Pressure Gradient px'"),
            (self.pgrad_ph[:,1] - self.pgrad_pI1[:,1], "Pressure Gradient py'"),
            
        ]):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            ax.plot_trisurf(x, y, data, cmap='viridis')
            ax.set_title(title)
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

from fealpy.sparse import csr_matrix
def sum_duplicates_csr_manual(csr):
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