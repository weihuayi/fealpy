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
    DirichletBC,
    StaggeredMeshManager
)


class StokesFVMStaggeredModel(ComputationalModel):
    """
    2D Stokes equation solver on staggered mesh using Finite Volume Method (FVM).
    This computational model first constructs a staggered mesh based on the initial mesh, where the initial mesh stores pressure, and the staggered mesh stores velocity.
    The momentum equations in different directions are discretized on their respective staggered meshes, the continuity equation is discretized on the initial mesh, and the resulting coupled linear system is solved directly to obtain all numerical solutions for the Stokes equation.

    Parameters:
        options (dict): Configuration dictionary for model setup.
            - 'pde': PDE data or index
            - 'nx', 'ny': mesh divisions
            - 'pbar_log', 'log_level': logging controls
    Attributes:
        mesh : The initialized computational mesh.
        pde : The PDE model object.
        uh,vh,ph : Numerical solution vector (computed after calling `solve`).
    
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "INFO"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_spaces()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh: {self.umesh.number_of_cells()} u-cells, "
            f"{self.vmesh.number_of_cells()} v-cells, "
            f"{self.pmesh.number_of_cells()} p-cells\n"
            f"  PDE: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[int, object]) -> None:
        if isinstance(pde, int):
            self.pde = PDEModelManager("stokes").get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, nx: int, ny: int) -> None:
        self.staggered_mesh = StaggeredMeshManager(self.pde, nx=nx, ny=ny)
        self.umesh = self.staggered_mesh.umesh
        self.vmesh = self.staggered_mesh.vmesh
        self.pmesh = self.staggered_mesh.pmesh
        self.UNC, self.VNC, self.PNC = (self.umesh.number_of_cells(),
                         self.vmesh.number_of_cells(),
                         self.pmesh.number_of_cells())
        self.h = 1.0 / nx

    def set_spaces(self) -> None:
        self.uspace = ScaledMonomialSpace2d(self.umesh, 0)
        self.vspace = ScaledMonomialSpace2d(self.vmesh, 0)


    def assemble_diffusion_u(self) -> Tuple[TensorLike, TensorLike]:
        """Assemble the diffusion matrix A on the u-mesh"""
        Au = BilinearForm(self.uspace).add_integrator(
            ScalarDiffusionIntegrator(q=2)).assembly()
        fu = LinearForm(self.uspace).add_integrator(
            ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        udbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                           threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        Au, fu = udbc.DiffusionApply(Au, fu)
        Au, fu = udbc.ThresholdApply(Au, fu)
        return Au, fu

    def assemble_diffusion_v(self) -> Tuple[TensorLike, TensorLike]:
        """Assemble the diffusion matrix B on the v-mesh"""
        Av = BilinearForm(self.vspace).add_integrator(
            ScalarDiffusionIntegrator(q=2)).assembly()
        fv = LinearForm(self.vspace).add_integrator(
            ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        vdbc = DirichletBC(self.vmesh, self.pde.dirichlet_velocity_v,
                           threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
        Av, fv = vdbc.DiffusionApply(Av, fv)
        Av, fv = vdbc.ThresholdApply(Av, fv)
        return Av, fv

    def assemble_pressure_gradient_u(self) -> TensorLike:
        """Assemble the pressure gradient matrix M1 on the u-mesh"""
        pedge_centers = self.pmesh.entity_barycenter('edge')
        ucell_centers = self.umesh.entity_barycenter('cell')
        dist1 = bm.sum((pedge_centers[:, None, :] - ucell_centers[None, :, :])**2, axis=2)
        ucell2pedge = bm.argmin(dist1, axis=0)
        pedge2cell = self.pmesh.edge_to_cell()
        upressurecell = pedge2cell[ucell2pedge, :2]
        M1 = COOTensor(
            indices=bm.stack([bm.repeat(bm.arange(self.UNC), 2), upressurecell.flatten()]),
            values=bm.tile([-self.h, self.h], self.UNC),
            spshape=(self.UNC, self.PNC)
        )
        return M1
    
    def assemble_pressure_gradient_v(self) -> TensorLike:
        """Assemble the pressure gradient matrix M2 on the v-mesh"""
        vcell_centers = self.vmesh.entity_barycenter('cell')
        pedge_centers = self.pmesh.entity_barycenter('edge')
        dist2 = bm.sum((pedge_centers[:, None, :] - vcell_centers[None, :, :])**2, axis=2)
        vcell2pedge = bm.argmin(dist2, axis=0)
        pedge2cell = self.pmesh.edge_to_cell()
        vpressurecell = pedge2cell[vcell2pedge, :2]
        M2 = COOTensor(
            indices=bm.stack([bm.repeat(bm.arange(self.VNC), 2), vpressurecell.flatten()]),
            values=bm.tile([-self.h, self.h], self.VNC),
            spshape=(self.VNC, self.PNC)
        )
        return M2

    def assemble_divergence_u(self) -> TensorLike:
        """Assemble the continuity equation matrix M3 (corresponding to u velocity) on the p-mesh"""
        uedge_centers = self.umesh.entity_barycenter('edge')
        pcell_centers = self.pmesh.entity_barycenter('cell')
        dist3 = bm.sum((uedge_centers[:, None, :] - pcell_centers[None, :, :])**2, axis=2)
        pcell2uedge = bm.argmin(dist3, axis=0)
        uedge2cell = self.umesh.edge_to_cell()
        uvelocitycell = uedge2cell[pcell2uedge, :2]
        M3 = COOTensor(
            indices=bm.stack([bm.repeat(bm.arange(self.PNC), 2), uvelocitycell.flatten()]),
            values=bm.tile([-self.h, self.h], self.PNC),
            spshape=(self.PNC, self.UNC)
        )
        return M3

    def assemble_divergence_v(self) -> TensorLike:
        """Assemble the continuity equation matrix M4 (corresponding to v velocity) on the p-mesh"""
        vedge_centers = self.vmesh.entity_barycenter('edge')
        pcell_centers = self.pmesh.entity_barycenter('cell')
        dist4 = bm.sum((vedge_centers[:, None, :] - pcell_centers[None, :, :])**2, axis=2)
        pcell2vedge = bm.argmin(dist4, axis=0)
        vedge2cell = self.vmesh.edge_to_cell()
        vvelocitycell = vedge2cell[pcell2vedge, :2]
        M4 = COOTensor(
            indices=bm.stack([bm.repeat(bm.arange(self.PNC), 2), vvelocitycell.flatten()]),
            values=bm.tile([-self.h, self.h], self.PNC),
            spshape=(self.PNC, self.VNC)
        )
        return M4

    def assemble_system(self) -> Tuple[TensorLike, TensorLike]:
        """Main assembly function to combine all matrix blocks into the complete linear system"""
        # Construct the block matrix
        # | A   0   M1^T |
        # | 0   B   M2^T |
        # | M3  M4   0   |
        # Here, M1^T and M2^T represent the discretized pressure gradients, and M3 and M4 represent the discretized divergence
        A,fu = self.assemble_diffusion_u()
        B,fv = self.assemble_diffusion_v()
        M1 = self.assemble_pressure_gradient_u()
        M2 = self.assemble_pressure_gradient_v()
        M3 = self.assemble_divergence_u()
        M4 = self.assemble_divergence_v()
        AB = BlockForm([[A, None], [None, B]]).assembly_sparse_matrix(format="csr")
        M5 = BlockForm([[M1], [M2]]).assembly_sparse_matrix(format="csr")
        M6 = BlockForm([[M3, M4]]).assembly_sparse_matrix(format="csr")
        ABC = BlockForm([[AB, M5], [M6, None]]).assembly_sparse_matrix(format="csr")

        LagA = self.pmesh.entity_measure("cell")
        A1 = COOTensor(
            bm.stack([bm.zeros(self.PNC, dtype=bm.int32),
                      bm.arange(self.UNC+self.VNC, self.UNC+self.VNC+self.PNC, dtype=bm.int32)], axis=0),
            LagA,
            spshape=(1, self.UNC+self.VNC+self.PNC)
        )
        b = bm.concatenate([fu, fv, bm.zeros(self.PNC), bm.array([self.pde.pressure_integral_target()])])
        S = BlockForm([[ABC, A1.T], [A1, None]]).assembly_sparse_matrix(format="csr")
        return S,b

    def solve(self) -> Tuple:
        S, b = self.assemble_system()
        sol = spsolve(S, b, "scipy")
        self.uh = sol[:self.UNC]
        self.vh = sol[self.UNC:self.UNC+self.VNC]
        self.ph = sol[self.UNC+self.VNC:-1]
        return self.uh, self.vh, self.ph

    def compute_error(self) -> Tuple:
        self.uI = self.pde.velocity_u(self.umesh.entity_barycenter("cell"))
        self.vI = self.pde.velocity_v(self.vmesh.entity_barycenter("cell"))
        self.pI = self.pde.pressure(self.pmesh.entity_barycenter("cell"))

        uerr = bm.sqrt(bm.sum(self.umesh.entity_measure("cell") * (self.uh - self.uI)**2))
        verr = bm.sqrt(bm.sum(self.vmesh.entity_measure("cell") * (self.vh - self.vI)**2))
        perr = bm.sqrt(bm.sum(self.pmesh.entity_measure("cell") * (self.ph - self.pI)**2))
        return uerr, verr, perr

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