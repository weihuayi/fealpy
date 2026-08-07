import numpy as np
from scipy.linalg import solve

from fealpy.backend import backend_manager as bm

from fealpy.csm.fem.hyperelastic_residual_integrator import (
    HyperElasticResidualIntegrator,
)
from fealpy.csm.fem.hyperelastic_tangent_integrator import (
    HyperElasticTangentIntegrator,
)


class HyperElasticLFEMModel:
    """
    Nonlinear LFEM model for displacement-controlled hyperelastic problems.

    This model is designed for the one-element Yeoh benchmark workflow:

        u -> Grad u -> F -> P(F), A(F) -> R_int, K -> Newton update

    Main residual convention:

        R(u, lambda) = R_int(u) - lambda * F_ext

    Dirichlet constraints are imposed directly on Newton increments:

        du[dof] = u_prescribed(lambda) - uh[dof]

    Parameters
    ----------
    space
        TensorFunctionSpace-like vector finite element space.
    material
        Hyperelastic material object with methods:
            stress(F), tangent(F), strain_energy_density(F).
    dbc : None, dict, list, tuple, ndarray
        Dirichlet boundary condition.

        Recommended form:
            dbc = {dof0: value0, dof1: value1, ...}

        If a list/array is given, those dofs are fixed to zero.
    force : None or array-like
        Global external load vector. For pure displacement-control benchmark,
        keep force=None.
    q : None or int
        Quadrature order.
    """

    def __init__(self, space, material, dbc=None, force=None, q=None):
        self.space = space
        self.mesh = space.mesh
        self.material = material
        self.dbc = dbc
        self.q = q

        self.residual_integrator = HyperElasticResidualIntegrator(
            material=material,
            space=space,
            q=q,
        )
        self.tangent_integrator = HyperElasticTangentIntegrator(
            material=material,
            space=space,
            q=q,
        )

        self.uh = space.function()
        self.gdof = space.number_of_global_dofs()

        if force is None:
            self.force = bm.zeros(self.gdof, dtype=self.uh.dtype)
        else:
            self.force = bm.array(force, dtype=self.uh.dtype)

        self.load_factor = 1.0

    # ------------------------------------------------------------------
    # Dirichlet boundary data
    # ------------------------------------------------------------------
    def get_dirichlet_data(self, load_factor=None):
        """
        Return fixed dofs and target values under current load factor.
        """
        if load_factor is None:
            load_factor = self.load_factor

        if self.dbc is None:
            return np.array([], dtype=np.int64), np.array([], dtype=float)

        if isinstance(self.dbc, dict):
            dofs = np.array(list(self.dbc.keys()), dtype=np.int64)
            values = np.array(list(self.dbc.values()), dtype=float)
            return dofs, load_factor * values

        dofs = np.array(self.dbc, dtype=np.int64)
        values = np.zeros(len(dofs), dtype=float)
        return dofs, values

    def enforce_dirichlet_value(self, load_factor=None):
        """
        Project the current displacement vector onto prescribed values.
        """
        dofs, values = self.get_dirichlet_data(load_factor)
        if len(dofs) > 0:
            self.uh[dofs] = values

    # ------------------------------------------------------------------
    # Kinematics and energy
    # ------------------------------------------------------------------
    def compute_F(self, uh=None):
        """
        Compute deformation gradient F = I + Grad_X u from displacement uh.

        Returns
        -------
        F : (NC, NQ, GD, GD)
        ws : (NQ,)
        cm : (NC,)
        """
        if uh is None:
            uh = self.uh

        space = self.space
        mesh = self.mesh
        scalar_space = getattr(space, "scalar_space", space)

        cell2dof = space.cell_to_dof()
        ue = uh[cell2dof]

        NC = ue.shape[0]
        GD = mesh.geo_dimension()

        q = scalar_space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        gphi = scalar_space.grad_basis(bcs, variable="x")
        ldof = gphi.shape[2]

        ue = ue.reshape(NC, GD, ldof)
        grad_u = bm.einsum("cia,cqaj->cqij", ue, gphi)
        F = grad_u + bm.eye(GD, dtype=grad_u.dtype)

        cm = mesh.entity_measure("cell")

        return F, ws, cm

    def compute_total_energy(self):
        """
        Compute total strain energy int W(F(u)) dV.
        """
        F, ws, cm = self.compute_F(self.uh)
        W = self.material.strain_energy_density(F)
        return bm.einsum("q,c,cq->", ws, cm, W)

    # ------------------------------------------------------------------
    # Local and global assembly
    # ------------------------------------------------------------------
    def residual(self):
        return self.residual_integrator.assembly_cell_vector(self.space, self.uh)

    def tangent(self):
        return self.tangent_integrator.assembly_cell_matrix(self.space, self.uh)

    def assemble_global_internal_residual(self):
        Re = self.residual()
        cell2dof = self.space.cell_to_dof()

        R = bm.zeros(self.gdof, dtype=Re.dtype)

        for c in range(Re.shape[0]):
            dofs = cell2dof[c]
            for a in range(Re.shape[1]):
                R[dofs[a]] += Re[c, a]

        return R

    def assemble_global_residual(self, load_factor=None):
        if load_factor is None:
            load_factor = self.load_factor

        Rint = self.assemble_global_internal_residual()
        return Rint - load_factor * self.force

    def assemble_global_tangent(self):
        Ke = self.tangent()
        cell2dof = self.space.cell_to_dof()

        K = bm.zeros((self.gdof, self.gdof), dtype=Ke.dtype)

        for c in range(Ke.shape[0]):
            dofs = cell2dof[c]
            for a in range(Ke.shape[1]):
                I = dofs[a]
                for b in range(Ke.shape[2]):
                    J = dofs[b]
                    K[I, J] += Ke[c, a, b]

        return K

    # ------------------------------------------------------------------
    # Boundary condition for Newton correction
    # ------------------------------------------------------------------
    def apply_dirichlet_bc_to_newton_system(self, K, R, load_factor=None):
        """
        Modify K and R for solving

            K du = -R

        so that fixed dofs satisfy

            du[d] = u_target[d] - uh[d].
        """
        dofs, values = self.get_dirichlet_data(load_factor)

        for d, v in zip(dofs, values):
            K[d, :] = 0.0
            K[:, d] = 0.0
            K[d, d] = 1.0
            R[d] = self.uh[d] - v

        return K, R

    def free_residual_norm(self, R, load_factor=None):
        dofs, _ = self.get_dirichlet_data(load_factor)

        R_np = np.asarray(R, dtype=float).copy()
        if len(dofs) > 0:
            R_np[dofs] = 0.0

        return np.linalg.norm(R_np)

    # ------------------------------------------------------------------
    # Line search
    # ------------------------------------------------------------------
    def line_search(
        self,
        du,
        load_factor=None,
        alpha0=1.0,
        reduction=0.5,
        max_ls=12,
    ):
        if load_factor is None:
            load_factor = self.load_factor

        uh_old = self.uh.copy()

        R0 = self.assemble_global_residual(load_factor)
        norm0 = self.free_residual_norm(R0, load_factor)
        E0 = float(self.compute_total_energy())

        alpha = alpha0

        for _ in range(max_ls):
            self.uh[:] = uh_old + alpha * du
            self.enforce_dirichlet_value(load_factor)

            R_trial = self.assemble_global_residual(load_factor)
            norm_trial = self.free_residual_norm(R_trial, load_factor)
            E_trial = float(self.compute_total_energy())

            if norm_trial <= (1.0 - 1.0e-4 * alpha) * norm0:
                return alpha

            if norm_trial < norm0 and E_trial <= E0 + 1.0e-12:
                return alpha

            alpha *= reduction

        self.uh[:] = uh_old
        self.enforce_dirichlet_value(load_factor)
        return 0.0

    # ------------------------------------------------------------------
    # Newton solver
    # ------------------------------------------------------------------
    def solve(
        self,
        tol=1.0e-8,
        maxit=25,
        nsteps=1,
        line_search=True,
        verbose=True,
    ):
        """
        Solve nonlinear equilibrium by load/displacement continuation.

        For displacement-control benchmark, give nonzero prescribed values in
        dbc, and set force=None. nsteps gradually scales those prescribed
        values from 0 to 1.
        """
        converged_all = True

        for step in range(1, nsteps + 1):
            load_factor = step / nsteps
            self.load_factor = load_factor
            self.enforce_dirichlet_value(load_factor)

            if verbose:
                print(f"\n==== load step {step}/{nsteps}, lambda = {load_factor:.6g} ====")

            converged = False

            for it in range(maxit):
                R = self.assemble_global_residual(load_factor)
                normR = self.free_residual_norm(R, load_factor)

                if verbose:
                    print(f"  iter {it:02d}, ||R_free|| = {normR:.12e}")

                if normR < tol:
                    converged = True
                    break

                K = self.assemble_global_tangent()
                K, Rbc = self.apply_dirichlet_bc_to_newton_system(
                    K,
                    R.copy(),
                    load_factor,
                )

                K_np = np.asarray(K, dtype=float)
                R_np = np.asarray(Rbc, dtype=float)

                du = solve(K_np, -R_np, assume_a="gen")

                if line_search:
                    alpha = self.line_search(du, load_factor)
                    if alpha == 0.0:
                        if verbose:
                            print("  line search failed, use alpha = 1e-3")
                        self.uh[:] = self.uh + 1.0e-3 * du
                        self.enforce_dirichlet_value(load_factor)
                    elif verbose:
                        print(f"           alpha = {alpha:.6g}")
                else:
                    self.uh[:] = self.uh + du
                    self.enforce_dirichlet_value(load_factor)

            if not converged:
                converged_all = False
                if verbose:
                    print("  WARNING: Newton did not converge in this load step.")
                break

        return converged_all, self.uh

    # ------------------------------------------------------------------
    # Postprocessing helper
    # ------------------------------------------------------------------
    def reaction_force(self, load_factor=None):
        """
        Return reaction vector on all dofs.

        For displacement-control benchmark, the reaction on constrained dofs is
        obtained from the unmodified global residual.
        """
        if load_factor is None:
            load_factor = self.load_factor
        return self.assemble_global_residual(load_factor)