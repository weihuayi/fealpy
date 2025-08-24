from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.solver import spsolve

from ..fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
)

class StokesFVMModel(ComputationalModel):
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
        self.cmv = bm.tile(self.cm, 2)

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

        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)
        A = B.diags()

        grad_p = GradientReconstruct(self.mesh).AverageGradientreNeumann(p, self.pde.neumann_pressure)  # (NC, 2)
        px = bm.transpose(grad_p).flatten()
        pin = bm.einsum('i,i->i', px, self.cmv)
        f = f - pin

        u = spsolve(B, f,"mumps")
        return A, u
    
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
    
    def pressure_correct(self, A: TensorLike, uf: TensorLike) -> TensorLike:
        """Solve for pressure correction p' to enforce continuity."""
        div_u = DivergenceReconstruct(self.mesh).Reconstruct(uf)  # (NE,)
        
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=2))
        A = bform2.assembly()
        LagA = self.mesh.entity_measure('cell')
        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                             bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b = bm.concatenate([div_u, b0], axis=0)
        sol = spsolve(A, b,"mumps")
        p_c = sol[:-1]
        return p_c

    def solve(self, max_iter: int = 10, tol: float = 1e-5) -> Tuple[TensorLike, TensorLike]:
        """Solve the Stokes equation using the SIMPLE algorithm."""
        p = bm.zeros(self.NC)
        A, u = self.temporary_velocity(p)
        self.residuals = []
        for i in range(max_iter):
            uf = self.Ucell2edge(u, self.pde.dirichlet_velocity)
            p_c = self.pressure_correct(A, uf)
            L2p_c = bm.sqrt(bm.sum(self.cm * (p_c)**2))
            self.residuals.append(float(L2p_c))
            self.logger.info(f"[Iter {i+1}] pressure_correct = {L2p_c:.4e}")
            if L2p_c < tol:
                self.logger.info("Converged.")
                break
            p = p - p_c
            A, u = self.temporary_velocity(p)

        self.uh = u
        self.ph = p
        return u, p

    def compute_error(self) -> Tuple[float, float]:
        """Compute L2 errors for velocity and pressure."""
        cell_centers = self.mesh.entity_barycenter('cell')
        u_exact = self.pde.velocity(cell_centers)
        u_exact = bm.transpose(u_exact).flatten()
        p_exact = self.pde.pressure(cell_centers)
        Verror = bm.sqrt(bm.sum(self.cmv * (self.uh - u_exact)**2))
        Perror = bm.sqrt(bm.sum(self.cm * (self.ph - p_exact)**2))
        return Verror, Perror

    def plot(self) -> None:
        """Plot numerical and exact solutions for u1, u2, and p."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        cell_centers = self.mesh.entity_barycenter('cell')
        x, y = cell_centers[:, 0], cell_centers[:, 1]
        u1, u2 = self.uh[:self.NC], self.uh[self.NC:]
        uI = self.pde.velocity(cell_centers)
        pI = self.pde.pressure(cell_centers)

        fig = plt.figure(figsize=(15, 10))
        titles = titles = [
            # ("Numerical u1", u1), ("Exact u1", uI[:, 0]),
            # ("Numerical u2", u2), ("Exact u2", uI[:, 1]),
            # ("Numerical p", self.ph), ("Exact p", pI),
            ("velocity error u'", u1- uI[:, 0]),
            ("velocity error v'", u2 -uI[:, 1]),
            ("Pressure error p'", self.ph - pI),
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