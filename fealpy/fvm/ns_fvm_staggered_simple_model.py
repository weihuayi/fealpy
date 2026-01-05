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
        p_edge = self.pmesh.entity_measure('edge')
        p_edge2 = bm.einsum('i,i->i', p_edge,p_edge)
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2,coef=p_edge / a_p_edge)
        ).assembly()  
        # print(1 / a_p_edge)p_edge
        # print(A.to_dense())
        # nbc = NeumannBC(self.pmesh, self.pde.neumann_pressure)
        # f = nbc.DiffusionApply(f)
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
        uf = bm.zeros(UNE)
        vf = bm.zeros(UNE)
        self.residuals = []

        vedge2uedge = self.staggered_mesh.get_dof_mapping_vedge2uedge()
        vf_umesh = vf[vedge2uedge]
        Uf = bm.stack([uf, vf_umesh], axis=1)

        uedge2vedge = self.staggered_mesh.get_dof_mapping_uedge2vedge()
        uf_vmesh = uf[uedge2vedge]
        Vf = bm.stack([uf_vmesh, vf], axis=1)
        ue2c = self.umesh.edge_to_cell()
        ve2c = self.vmesh.edge_to_cell()
        for i in range(max_iter):

            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p)
            uh, a_p_u = self.compute_temporary_velocity_u(p_u,Uf)
            vh, a_p_v = self.compute_temporary_velocity_v(p_v,Vf)
            
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
            self.div_rhs = self.div.StagReconstruct(edge_vel)
            p_corr = self.correct_pressure_compute(-self.div_rhs, a_p_edge)
            err = bm.sqrt(bm.sum(self.pcm * p_corr ** 2))
            self.residuals.append(float(err))
            self.logger.info(f"[Iter {i+1}] Pressure correction residual: {err:.2e}")
            if err < tol:
                self.logger.info("Converged.")
                break
            ucell2pedge = self.staggered_mesh.get_dof_mapping_ucell2pedge()
            vcell2pedge = self.staggered_mesh.get_dof_mapping_vcell2pedge()
            pe2c = self.pmesh.edge_to_cell()
            u_corr = (p_corr[pe2c[ucell2pedge,0]]-p_corr[pe2c[ucell2pedge,1]])/self.staggered_mesh.hx
            v_corr = (p_corr[pe2c[vcell2pedge,0]]-p_corr[pe2c[vcell2pedge,1]])/self.staggered_mesh.hy
            u_corr = self.ucm / a_p_u * u_corr
            v_corr = self.vcm / a_p_v * v_corr
            # if err < 1e-3:
            #     p += 0.3*p_corr
            # elif err < 1e-1:
            #     p += 0.05*p_corr
            # else:
            #     p += 0.05*p_corr
            p += 0.25*p_corr
            uh += u_corr
            vh += v_corr
            uf1 = (uh[ue2c[:,0]] + uh[ue2c[:,1]])/2
            vf1 = (vh[ve2c[:,0]] + vh[ve2c[:,1]])/2
            vf_umesh = vf1[vedge2uedge]
            Uf = bm.stack([uf1, vf_umesh], axis=1)
            uf_vmesh = uf1[uedge2vedge]
            Vf = bm.stack([uf_vmesh, vf1], axis=1)

        
        self.uh, self.vh, self.ph = uh, vh, p
        self.edge_vel, _ = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
        # print(self.edge_vel.shape)
        # print(self.pmesh.number_of_edges())
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

    def plot_streamline(self) -> None:
        """Plot the streamlines of the velocity field."""
        import matplotlib.pyplot as plt

        nx, ny = 32, 32 # u.shape = (ny, nx) 通常
        
        c2e = self.pmesh.cell2edge
        u = (self.edge_vel[c2e[:,1]]+self.edge_vel[c2e[:,3]])/2
        v = (self.edge_vel[c2e[:,0]]+self.edge_vel[c2e[:,2]])/2
        print("u shape:",u.shape)
        print("v shape:",v.shape)
        u2d = u.reshape((ny, nx),order="F")
        v2d = v.reshape((ny, nx),order="F")
        p2d = self.ph.reshape((ny, nx),order="F")

        import numpy as np

        dx = 1.0 / nx
        dy = 1.0 / ny

        x = (np.arange(nx) + 0.5) * dx
        y = (np.arange(ny) + 0.5) * dy
        X, Y = np.meshgrid(x, y)


        import matplotlib.pyplot as plt

        speed = np.sqrt(u2d**2 + v2d**2)

        plt.figure(figsize=(6, 6))
        plt.streamplot(
            X, Y, u2d, v2d,
            color=speed,
            cmap="viridis",
            density=1.5
        )
        plt.colorbar(label="|u|")
        plt.axis("equal")
        plt.title("Lid-driven cavity streamlines")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, p2d, levels=50, cmap="coolwarm")
        plt.colorbar(label="p")
        plt.axis("equal")
        plt.title("Pressure contour")
        plt.show()
