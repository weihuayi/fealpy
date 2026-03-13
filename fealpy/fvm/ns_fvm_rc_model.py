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
    GradientReconstruct,
    DirichletBC,
    NeumannBC,
    ConvectionIntegrator
)


class NSFVMRCModel(ComputationalModel):
    """
    A 2D NS equation solver using the finite volume method (FVM).

    This computational model solves the 2D NS equation on a uniform grid, 
    incorporating Rhie-Chow interpolation correction to mitigate oscillations in the numerical pressure solution.

    Parameters:
        options (dict): Configuration dictionary for model setup.
            - 'pde': PDE data or index
            - 'nx', 'ny': mesh divisions
            - 'pbar_log', 'log_level': logging controls

    Attributes:
        mesh : The initialized computational mesh.
        pde : The PDE model object.
        uh,vh,ph : Numerical solution vector (computed after calling `solve_rhie_chow`).
    
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "INFO"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space()


    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh: {self.mesh.number_of_cells()} cells\n"
            f"  Space degree: {self.p}\n"
            f"  PDE: {type(self.pde).__name__}\n"
        )
    
    def set_pde(self, pde: Union[str, object]) -> None:
        if isinstance(pde, int):
            self.pde = PDEModelManager('navier_stokes').get_example(pde)
        else:
            self.pde = pde

        self.logger.info(self.pde)

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        self.mesh = self.pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)
        self.NC = self.mesh.number_of_cells()
        self.h = 1/nx   

    def set_space(self, degree: int = 0) -> None:
        self.p = degree
        self.pspace = ScaledMonomialSpace2d(self.mesh, self.p)
        self.uspace = TensorFunctionSpace(self.pspace, shape=(2, -1))

    def assembly_velocity(self,uf) -> Tuple[TensorLike, TensorLike]:
        """
        Discretize the velocity term
        """
        bform = BilinearForm(self.uspace).add_integrator(
            ScalarDiffusionIntegrator(q=2))
        bform.add_integrator(ConvectionIntegrator(q=2,coef=uf))
        AB = bform.assembly()
        # AB = BilinearForm(self.uspace).add_integrator(
        #     ScalarDiffusionIntegrator(q=2)).assembly()

        f = LinearForm(self.uspace).add_integrator(
            ScalarSourceIntegrator(self.pde.source, q=2)).assembly()
    
        return AB, f

    def assembly_pressure(self) -> Tuple[TensorLike, TensorLike]:
        """
        Discretize the pressure term
        """
        NE = self.mesh.number_of_edges()
        c = bm.tile([1, 0], (NE, 1))
        d = bm.tile([0, 1], (NE, 1))

        M1 = BilinearForm(self.pspace).add_integrator(
            ConvectionIntegrator(q=2, coef=c)).assembly()
        M2 = BilinearForm(self.pspace).add_integrator(
            ConvectionIntegrator(q=2, coef=d)).assembly()

        return M1, M2

    def lagrange_multiplier(self) -> TensorLike:
        """
        Ensure the uniqueness of the numerical pressure solution
        under pure Neumann boundary conditions using the Lagrange multiplier method
        """
        LagA = self.mesh.entity_measure('cell')
        A1 = COOTensor(
            bm.stack([
                bm.zeros(len(LagA), dtype=bm.int32),
                bm.arange(2 * len(LagA), 3 * len(LagA), dtype=bm.int32)
            ], axis=0),
            LagA, spshape=(1, 3 * len(LagA))
        )
        return A1
    
    def assembly_base_system(self,uf=None) -> Tuple:
        """
        Apply boundary conditions to the discretized velocity 
        and pressure terms, and assemble them into basic matrix blocks using BlockForm
        """
        AB, f = self.assembly_velocity(uf)
        M1, M2 = self.assembly_pressure()
        M3 = BlockForm([[M1, M2]]).assembly_sparse_matrix(format='csr')
        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        nbc = NeumannBC(self.mesh, self.pde.neumann_pressure)
        AB, f = dbc.DiffusionApply(AB, f)
        ap = AB.diags().values
        # f[:self.NC] = dbc.ConvectionApplyX(f[:self.NC],self.pde.dirichlet_velocity)
        # f[self.NC:] = dbc.ConvectionApplyX(f[self.NC:],self.pde.dirichlet_velocity)
        M1 = nbc.ConvectionApplyX(M1, f[:self.NC])
        M2 = nbc.ConvectionApplyY(M2, f[self.NC:])
        
        M4 = BlockForm([[M1], [M2]]).assembly_sparse_matrix(format='csr')

        return AB, M3, M4, f, ap
    
    def assembly_rhie_chow_corrected_system(self, ph0, ap) -> Tuple[TensorLike, TensorLike]:
        """
        Implement Rhie-Chow interpolation correction; 
        the assembled matrix M5 corresponds to the discretization of the true gradient of pressure p at control volume edges, 
        while rc corresponds to the gradient of pressure p obtained by direct interpolation at control volume edges
        """
        NC = self.mesh.number_of_cells()
        e2c = self.mesh.edge_to_cell()
        Sf = self.mesh.edge_normal()
        ap_edge = (ap[e2c[:, 0]] + ap[e2c[:, 1]]) / 2
        # grad_f1 = (ph0[e2c[:, 1]] - ph0[e2c[:, 0]]) / self.h
        grad_p = GradientReconstruct(self.mesh).AverageGradientreNeumann(ph0, self.pde.neumann_pressure)
        grad_f2 = GradientReconstruct(self.mesh).reconstruct(grad_p)

        x = self.mesh.boundary_face_index()
        mask = bm.ones(grad_f2.shape[0], dtype=bool)
        mask[x] = False
        # r1 = bm.einsum('i,i->i', ap_edge, grad_f1)
        r2 = bm.einsum('i,ij->ij', ap_edge, grad_f2)
        c = bm.einsum('ij,ij->i', Sf, r2)
        rc = bm.zeros(NC)
        bm.add.at(rc, e2c[mask, 0], c[mask])
        bm.add.at(rc, e2c[mask, 1], -c[mask])
        bdu = self.pde.dirichlet_velocity(self.mesh.entity_barycenter('edge')[x])
        d = bm.einsum('ij,ij->i', bdu, Sf[x])
        bm.add.at(rc, e2c[x, 0], d)
        M5 = BilinearForm(self.pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2, coef=ap_edge)).assembly()

        return M5,rc


    def solve_rhie_chow(self, max_iter: int = 1, tol: float = 1e-7) -> Tuple:
        # Sf = self.mesh.edge_normal()
        # Uf = bm.stack([bm.ones_like(Sf[:,0]), bm.zeros_like(Sf[:,0])], axis=1)
        self.uI = self.pde.velocity_u(self.mesh.entity_barycenter("edge"))
        self.vI = self.pde.velocity_v(self.mesh.entity_barycenter("edge"))
        Uf = bm.stack([self.uI, self.vI], axis=1)
        ph = bm.zeros(self.NC)
        e2c = self.mesh.edge_to_cell()
        for i in range(max_iter):
            AB, M3, M4, f, ap = self.assembly_base_system(Uf)
            A1 = self.lagrange_multiplier()
            ABC = BlockForm([[AB, M4], [M3, None]]).assembly_sparse_matrix(format='csr')
            S = BlockForm([[ABC, A1.T], [A1, None]]).assembly_sparse_matrix(format='csr')
            b0 = bm.array([self.pde.pressure_integral_target()])
            b = bm.concatenate([f,bm.zeros(self.NC),b0], axis=0)
            sol = spsolve(S, b, "mumps")
            ph0 = sol[2 * self.NC:-1]
            # uh = sol[:self.NC]
            # vh = sol[self.NC:2*self.NC]
            # ph = sol[2*self.NC:-1]

            M5,rc = self.assembly_rhie_chow_corrected_system(ph0, ap)
            AB2 = BlockForm([[AB,M4],[M3,M5]]).assembly_sparse_matrix(format='csr')
            S2 = BlockForm([[AB2, A1.T], [A1, None]])
            S2 = S2.assembly_sparse_matrix(format='csr')
            b2 = bm.concatenate([f,-rc,b0], axis=0)
            
            sol = spsolve(S2, b2, "mumps")
            uh = sol[:self.NC]
            vh = sol[self.NC:2*self.NC]
            ph = sol[2*self.NC:-1]
            uf1 = (uh[e2c[:,0]] + uh[e2c[:,1]])/2
            vf1 = (vh[e2c[:,0]] + vh[e2c[:,1]])/2
            Uf1 = bm.stack([uf1,vf1],axis=1)
            res = bm.max(bm.abs(Uf1[:,0] - Uf[:,0]))
            Uf = Uf1
            self.logger.info(f"Iteration {i+1}, Residual: {res:.6e}")
            if res < tol:
                break
        self.uh, self.vh, self.ph, self.ph0 = uh, vh, ph, ph0
        return self.uh, self.vh, self.ph
        
    
    def compute_error(self) -> Tuple:
        """
        Compute the error between the numerical solutions for velocity u, v, 
        and pressure p and their analytical solutions
        """
        self.uI = self.pde.velocity_u(self.mesh.entity_barycenter("cell"))
        self.vI = self.pde.velocity_v(self.mesh.entity_barycenter("cell"))
        self.pI = self.pde.pressure(self.mesh.entity_barycenter("cell"))
        uerr = bm.max(bm.abs(self.uh - self.uI))
        verr = bm.max(bm.abs(self.vh - self.vI))
        perr = bm.max(bm.abs(self.ph - self.pI))
        # uerr = bm.sqrt(bm.sum(self.mesh.entity_measure("cell") * (self.uh - self.uI)**2))
        # verr = bm.sqrt(bm.sum(self.mesh.entity_measure("cell") * (self.vh - self.vI)**2))
        # perr = bm.sqrt(bm.sum(self.mesh.entity_measure("cell") * (self.ph - self.pI)**2))
        return uerr, verr, perr
    

    def plot(self) -> None:
        """
        Plot the error of the numerical solutions for velocity u, v, and pressure p
        """
        import matplotlib.pyplot as plt
        ppoints = self.mesh.entity_barycenter('cell')
        x, y = ppoints[:, 0], ppoints[:, 1]
        fig = plt.figure(figsize=(12, 8))
        for i, (data, title) in enumerate([
                (self.uh - self.uI, "Error u'"),
                (self.vh - self.vI, "Error v'"),
                (self.ph - self.pI, " Error p(RC)'"),
                (self.ph0 - self.pI, " Error p(non-RC)"),
                ]):
                ax = fig.add_subplot(2, 3, i+1, projection='3d')
                ax.plot_trisurf(x, y, data, cmap='viridis')
                ax.set_title(title)
        plt.tight_layout()
        plt.show()