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
    ScalarCrossDiffusionIntegrator,
    ScalarSourceIntegrator,
    FVMGeometry,
    GradientReconstruct,
    reconstruct_face_gradient,
    ConvectionIntegrator,
    cell_average_l2_error,
)
from .legacy_boundary_conditions import (
    ExperimentalDirichletBC as DirichletBC,
    ExperimentalNeumannBC as NeumannBC,
)
from .rhie_chow_coupled_operator import RhieChowCoupledOperator


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
        self.mesh_type = self.options.get("mesh_type", "uniform_qrad")
        self.mesh = self.pde.init_mesh[self.mesh_type](nx=nx, ny=ny)
        self.NC = self.mesh.number_of_cells()
        self.h = 1/nx   

    def set_space(self, degree: int = 0) -> None:
        self.p = degree
        self.pspace = ScaledMonomialSpace2d(self.mesh, self.p)
        self.uspace = TensorFunctionSpace(self.pspace, shape=(2, -1))
        self.pressure_gradient = GradientReconstruct(self.mesh)
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            method="green_gauss",
            boundary_value=self.pde.dirichlet_velocity,
            boundary_type="dirichlet",
        )
        self.fvm_geometry = FVMGeometry(self.mesh)
        self.velocity_dirichlet_bc = DirichletBC(
            self.mesh, self.pde.dirichlet_velocity
        )
        self.rc_operator = RhieChowCoupledOperator(self.mesh)

    def assembly_velocity(self, uf, u0=None) -> Tuple[TensorLike, TensorLike]:
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
        f = self.velocity_dirichlet_bc.apply_convection(
            f,
            uf,
            components=self.GD,
        )
        if u0 is not None:
            f = f + self.compute_cross_diffusion(u0)
    
        return AB, f

    def compute_cross_diffusion(self, uh: TensorLike) -> TensorLike:
        """Assemble the explicit non-orthogonal diffusion correction."""
        lform = LinearForm(self.uspace)
        U = bm.stack((uh[:self.NC], uh[self.NC:]), axis=1)
        grad_u = self.velocity_gradient.cell_gradient(U)
        grad_f = reconstruct_face_gradient(self.mesh, grad_u)
        lform.add_integrator(
            ScalarCrossDiffusionIntegrator(
                uh,
                grad_f,
                geometry=self.fvm_geometry,
                method="bounded_over_relaxed",
                boundary_policy="all",
            )
        )
        return lform.assembly()

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
    
    def assembly_base_system(self, uf=None, u0=None) -> Tuple:
        """
        Apply boundary conditions to the discretized velocity 
        and pressure terms, and assemble them into basic matrix blocks using BlockForm
        """
        AB, f = self.assembly_velocity(uf, u0)
        M1, M2 = self.assembly_pressure()
        M3 = BlockForm([[M1, M2]]).assembly_sparse_matrix(format='csr')
        nbc = NeumannBC(self.mesh, self.pde.neumann_pressure)
        AB, f = self.velocity_dirichlet_bc.apply_diffusion(
            AB,
            f,
            components=self.GD,
        )
        ap = self._matrix_diagonal(AB)
        if callable(getattr(self.pde, "pressure_dirichlet", None)):
            f = f - self._pressure_boundary_force()
        else:
            M1 = nbc.ConvectionApplyX(M1, f[:self.NC])
            M2 = nbc.ConvectionApplyY(M2, f[self.NC:])
        
        M4 = BlockForm([[M1], [M2]]).assembly_sparse_matrix(format='csr')

        return AB, M3, M4, f, ap

    def _pressure_boundary_force(self):
        bd_face = self.mesh.boundary_face_index()
        owner = self.mesh.edge_to_cell()[bd_face, 0]
        face_center = self.mesh.entity_barycenter('face')[bd_face]
        pressure = self.pde.pressure_dirichlet(face_center)
        Sf = self.mesh.edge_normal()[bd_face]
        fx = bm.zeros(self.NC, dtype=pressure.dtype)
        fy = bm.zeros(self.NC, dtype=pressure.dtype)
        fx = bm.index_add(fx, owner, pressure * Sf[:, 0], axis=0)
        fy = bm.index_add(fy, owner, pressure * Sf[:, 1], axis=0)
        return bm.concatenate([fx, fy], axis=0)

    def _matrix_diagonal(self, A):
        diag_entries = A.diags()
        diag = bm.zeros(A.shape[0], dtype=diag_entries.values.dtype)
        diag = bm.index_add(diag, diag_entries.indices, diag_entries.values, axis=0)
        return diag

    def _relative_inf_norm(self, delta, reference):
        denominator = bm.maximum(1.0, bm.max(bm.abs(reference)))
        return bm.max(bm.abs(delta)) / denominator

    def _pressure_delta_without_mean(self, p_new, p_old):
        delta = p_new - p_old
        cell_measure = self.mesh.entity_measure("cell")
        mean_delta = bm.sum(cell_measure * delta) / bm.sum(cell_measure)
        return delta - mean_delta

    def _rc_rhs(self, rc_operator, ap, pressure):
        _, bp = rc_operator.assemble_pressure_block(ap, p_old=pressure)
        bd_face = self.mesh.boundary_face_index()
        bd_point = self.mesh.entity_barycenter("edge")[bd_face]
        bd_velocity = self.pde.dirichlet_velocity(bd_point)
        return bp + rc_operator.boundary_velocity_rhs(bd_velocity)

    def _aitken_omega(self, delta, state, omega_min, omega_max):
        previous_delta = state.get("previous_delta")
        previous_omega = state.get("omega", 1.0)
        if previous_delta is None:
            return previous_omega

        delta2 = delta - previous_delta
        denominator = bm.sum(delta2 * delta2)
        if float(denominator) <= 1.0e-30:
            return previous_omega

        numerator = bm.sum(previous_delta * delta2)
        omega = -previous_omega * numerator / denominator
        omega = bm.minimum(omega_max, bm.maximum(omega_min, omega))
        return float(omega)

    def _relax_pressure(self, p_old, p_raw, state, *, relaxation, omega, omega_min, omega_max):
        delta = p_raw - p_old
        if relaxation == "picard":
            used_omega = 1.0
        elif relaxation == "fixed":
            used_omega = omega
        elif relaxation == "aitken":
            used_omega = self._aitken_omega(delta, state, omega_min, omega_max)
        else:
            raise ValueError("relaxation must be 'picard', 'fixed', or 'aitken'.")

        p_used = p_old + used_omega * delta
        return p_used, {"previous_delta": delta, "omega": used_omega}
    
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
        grad_p = self.pressure_gradient.cell_gradient(ph0)
        grad_f2 = reconstruct_face_gradient(self.mesh, grad_p)

        x = self.mesh.boundary_face_index()
        mask = bm.ones(grad_f2.shape[0], dtype=bm.bool)
        mask = bm.set_at(mask, x, False)
        # r1 = bm.einsum('i,i->i', ap_edge, grad_f1)
        r2 = bm.einsum('i,ij->ij', ap_edge, grad_f2)
        c = bm.einsum('ij,ij->i', Sf, r2)
        rc = bm.zeros(NC, dtype=c.dtype)
        rc = bm.index_add(rc, e2c[mask, 0], c[mask], axis=0)
        rc = bm.index_add(rc, e2c[mask, 1], c[mask], axis=0, alpha=-1)
        bdu = self.pde.dirichlet_velocity(self.mesh.entity_barycenter('edge')[x])
        d = bm.einsum('ij,ij->i', bdu, Sf[x])
        rc = bm.index_add(rc, e2c[x, 0], d, axis=0)
        M5 = BilinearForm(self.pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2, coef=ap_edge)).assembly()

        return M5,rc

    def assembly_rhie_chow_pressure_block(self, ap, p_old=None) -> Tuple[TensorLike, TensorLike]:
        """
        Assemble the Rhie-Chow pressure block directly for the coupled system.

        Unlike ``assembly_rhie_chow_corrected_system``, this path does not use
        an unstabilized pressure solution to build the correction.  The compact
        pressure-pressure block is assembled from the actual momentum diagonal.
        """
        return self.rc_operator.assemble_pressure_block(ap, p_old)


    def solve_rhie_chow(
        self,
        max_iter: int = 50,
        min_iter: int = 2,
        tol: float = 1e-7,
        relaxation: str = "picard",
        omega: float = 1.0,
        omega_min: float = 0.2,
        omega_max: float = 1.2,
        return_diagnostics: bool = False,
    ) -> Tuple:
        # Sf = self.mesh.edge_normal()
        # Uf = bm.stack([bm.ones_like(Sf[:,0]), bm.zeros_like(Sf[:,0])], axis=1)
        self.uI = self.pde.velocity_u(self.mesh.entity_barycenter("edge"))
        self.vI = self.pde.velocity_v(self.mesh.entity_barycenter("edge"))
        Uf = bm.stack([self.uI, self.vI], axis=1)
        u0 = bm.zeros(2 * self.NC)
        ph = bm.zeros(self.NC)
        diagnostics = []
        relaxation_state = {
            "previous_delta": None,
            "omega": omega,
        }
        for i in range(max_iter):
            AB, M3, M4, f, ap = self.assembly_base_system(Uf, u0)
            A1 = self.lagrange_multiplier()
            b0 = bm.array([self.pde.pressure_integral_target()])
            rc_operator = self.rc_operator
            LRC = rc_operator.pressure_stabilization_matrix(ap)
            rc_rhs_old = self._rc_rhs(rc_operator, ap, ph)
            bd_face = self.mesh.boundary_face_index()
            bd_velocity = self.pde.dirichlet_velocity(self.mesh.entity_barycenter('edge')[bd_face])
            AB2 = BlockForm([[AB, M4], [M3, LRC]]).assembly_sparse_matrix(format='csr')
            S2 = BlockForm([[AB2, A1.T], [A1, None]])
            S2 = S2.assembly_sparse_matrix(format='csr')
            b2 = bm.concatenate([f, rc_rhs_old, b0], axis=0)
            
            sol = spsolve(S2, b2, "mumps")
            uh = sol[:self.NC]
            vh = sol[self.NC:2*self.NC]
            ph_raw = sol[2*self.NC:-1]
            ph_next, relaxation_state = self._relax_pressure(
                ph,
                ph_raw,
                relaxation_state,
                relaxation=relaxation,
                omega=omega,
                omega_min=omega_min,
                omega_max=omega_max,
            )
            velocity = bm.stack([uh, vh], axis=1)
            Uf1 = rc_operator.face_velocity(velocity, ap, ph_next)
            Uf1 = bm.set_at(Uf1, bd_face, bd_velocity)
            pressure_delta = self._pressure_delta_without_mean(ph_next, ph)
            rc_rhs_new = self._rc_rhs(rc_operator, ap, ph_next)
            res_u = self._relative_inf_norm(Uf1 - Uf, Uf1)
            res_p = self._relative_inf_norm(pressure_delta, ph_next)
            res_b = self._relative_inf_norm(rc_rhs_new - rc_rhs_old, rc_rhs_new)
            diagnostics.append({
                "iteration": i + 1,
                "res_face_velocity": float(res_u),
                "res_pressure": float(res_p),
                "res_rc_rhs": float(res_b),
                "omega": float(relaxation_state["omega"]),
            })
            Uf = Uf1
            ph = ph_next
            u0 = bm.concatenate([uh, vh], axis=0)
            self.logger.info(
                f"Iteration {i+1}, Residuals: "
                f"Uf={float(res_u):.6e}, p={float(res_p):.6e}, "
                f"rc_rhs={float(res_b):.6e}"
            )
            converged = res_u < tol and res_p < tol and res_b < tol
            if i + 1 >= min_iter and converged:
                break
        self.uh, self.vh, self.ph = uh, vh, ph
        self.rhie_chow_diagnostics = diagnostics
        if return_diagnostics:
            return self.uh, self.vh, self.ph, diagnostics
        return self.uh, self.vh, self.ph
        
    
    def compute_error(self) -> Tuple:
        """
        Compute the error between the numerical solutions for velocity u, v, 
        and pressure p and their analytical solutions
        """
        q = getattr(self, "error_quadrature_order", 4)
        uerr, self.uI = cell_average_l2_error(self.mesh, self.pde.velocity_u, self.uh, q=q)
        verr, self.vI = cell_average_l2_error(self.mesh, self.pde.velocity_v, self.vh, q=q)
        perr, self.pI = cell_average_l2_error(self.mesh, self.pde.pressure, self.ph, q=q)
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
                ]):
                ax = fig.add_subplot(2, 3, i+1, projection='3d')
                ax.plot_trisurf(x, y, data, cmap='viridis')
                ax.set_title(title)
        plt.tight_layout()
        plt.show()
