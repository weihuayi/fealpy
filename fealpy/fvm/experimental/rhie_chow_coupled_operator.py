"""Experimental coupled Rhie-Chow pressure operator."""

from fealpy.backend import backend_manager as bm
from fealpy.sparse import COOTensor

from ..face_gradient import reconstruct_face_gradient
from ..fvm_geometry import FVMGeometry


class RhieChowCoupledOperator:
    """Experimental Rhie-Chow pressure operator for coupled RC solvers.

    This class assembles the compact pressure-pressure block produced by
    substituting Rhie-Chow face interpolation into the continuity equation.
    It is intended for coupled systems of the form ``[[A, G], [B, LRC]]``.

    It is not the stable SIMPLE/PISO face-velocity interpolation API.  The
    production collocated SIMPLE path uses ``RhieChowInterpolation`` in
    ``fealpy.fvm.collocated_face_velocity_reconstruct``.
    """

    def __init__(self, mesh, rho=1.0):
        from ..gradient_reconstruct import GradientReconstruct

        self.mesh = mesh
        self.rho = rho
        self.gradient_reconstruct = GradientReconstruct(mesh)
        self.fvm_geometry = FVMGeometry(mesh)
        self._cell_lsq_gradient_matrix_cache = None

    def pressure_stabilization_matrix(self, ap):
        """
        Assemble the compact Rhie-Chow pressure block LRC.

        Parameters
        ----------
        ap
            Momentum diagonal coefficients. For a 2D coupled velocity block,
            the expected layout is ``[a_u, a_v]`` with length ``2*NC``.
            A scalar-cell array of length ``NC`` is also accepted and used for
            both velocity components.
        """
        NC = self.mesh.number_of_cells()
        owner, neighbour, _, _, _, beta = self._internal_pressure_flux_geometry(ap)
        alpha = self.rho * beta
        row = bm.concat([owner, owner, neighbour, neighbour])
        col = bm.concat([owner, neighbour, neighbour, owner])
        values = bm.concat([alpha, -alpha, alpha, -alpha])
        indices = bm.stack([row, col], axis=0)
        return COOTensor(indices, values, spshape=(NC, NC)).tocsr()

    def explicit_pressure_rhs(self, ap, p_old=None):
        """
        Assemble the explicit wide-stencil pressure-gradient correction.

        Passing ``p_old=None`` returns a zero vector.  Otherwise ``p_old`` is
        used to reconstruct the pressure-gradient contribution from the same
        non-orthogonal face split as :meth:`pressure_stabilization_matrix`.
        """
        NC = self.mesh.number_of_cells()
        if p_old is None:
            return bm.zeros(NC)

        owner, neighbour, is_internal, d_pf, _, beta = (
            self._internal_pressure_flux_geometry(ap)
        )
        grad_p = self.gradient_reconstruct.cell_gradient(p_old)
        grad_f = reconstruct_face_gradient(self.mesh, grad_p)[is_internal]
        d_dot_grad = bm.einsum("ij,ij->i", d_pf, grad_f)
        face_rhs = -self.rho * beta * d_dot_grad

        rhs = bm.zeros(NC, dtype=face_rhs.dtype)
        rhs = bm.index_add(rhs, owner, face_rhs, axis=0)
        rhs = bm.index_add(rhs, neighbour, face_rhs, axis=0, alpha=-1)
        return rhs

    def explicit_pressure_matrix(self, ap):
        """
        Assemble the matrix form of ``explicit_pressure_rhs(ap, p)``.

        The wide-stencil Rhie-Chow pressure-gradient contribution is linear in
        pressure.  For steady Stokes coupled solves it should be placed on the
        left-hand side instead of being updated by a Picard loop.
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        owner, neighbour, _, d_pf, _, beta = self._internal_pressure_flux_geometry(ap)
        n_internal = owner.shape[0]

        coeff_x = -self.rho * beta * d_pf[:, 0]
        coeff_y = -self.rho * beta * d_pf[:, 1]

        face = bm.arange(n_internal, dtype=owner.dtype)
        face_rows = bm.concat([face, face, face, face])
        face_cols = bm.concat([
            2 * owner,
            2 * neighbour,
            2 * owner + 1,
            2 * neighbour + 1,
        ])
        face_values = bm.concat([
            0.5 * coeff_x,
            0.5 * coeff_x,
            0.5 * coeff_y,
            0.5 * coeff_y,
        ])
        face_gradient = COOTensor(
            bm.stack([face_rows, face_cols], axis=0),
            face_values,
            spshape=(n_internal, 2 * NC),
        ).coalesce().tocsr()

        scatter_rows = bm.concat([owner, neighbour])
        scatter_cols = bm.concat([face, face])
        one = bm.ones(n_internal, dtype=face_values.dtype)
        scatter_values = bm.concat([one, -one])
        scatter = COOTensor(
            bm.stack([scatter_rows, scatter_cols], axis=0),
            scatter_values,
            spshape=(NC, n_internal),
        ).coalesce().tocsr()

        return scatter @ face_gradient @ self._cell_lsq_gradient_matrix()

    def assemble_pressure_block(self, ap, p_old=None):
        """Return ``(LRC, bp)`` for the coupled continuity equation."""
        return self.pressure_stabilization_matrix(ap), self.explicit_pressure_rhs(ap, p_old)

    def _cell_lsq_gradient_matrix(self):
        """Return the private cell-LSQ gradient matrix used by the RC block.

        This is the matrix form of the unweighted two-layer LSQ gradient used
        to linearize ``explicit_pressure_rhs(ap, p)``.  It intentionally has no
        boundary-condition handling and should not be treated as the general
        gradient reconstruction API.
        """
        if self._cell_lsq_gradient_matrix_cache is not None:
            return self._cell_lsq_gradient_matrix_cache

        mesh = self.mesh
        NC = mesh.number_of_cells()
        c2c = mesh.cell_to_cell()
        N = bm.concatenate((c2c[c2c].reshape(NC, -1), c2c), axis=1)
        N_sorted = bm.sort(N, axis=1)
        dup_mask = bm.zeros_like(N_sorted, dtype=bool)
        dup_mask = bm.set_at(
            dup_mask,
            (slice(None), slice(1, None)),
            N_sorted[:, 1:] == N_sorted[:, :-1],
        )
        row_broadcast = bm.broadcast_to(
            bm.arange(N.shape[0], dtype=N_sorted.dtype)[:, None], N_sorted.shape
        )
        N_unique = bm.copy(N_sorted)
        N_unique = bm.set_at(N_unique, dup_mask, row_broadcast[dup_mask])
        N = bm.sort(N_unique, axis=1)

        cell_centers = mesh.entity_barycenter("cell")
        d = cell_centers[N] - cell_centers[:, None, :]
        A = bm.sum(bm.einsum("hij,hik->hijk", d, d), axis=1)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        wx = (A[:, 1, 1, None] * d[..., 0] - A[:, 0, 1, None] * d[..., 1]) / det[:, None]
        wy = (-A[:, 1, 0, None] * d[..., 0] + A[:, 0, 0, None] * d[..., 1]) / det[:, None]

        cell = bm.arange(NC, dtype=N.dtype)
        base_rows = bm.repeat(cell, N.shape[1])
        neighbour_cols = N.reshape(-1)
        rows = bm.concat([
            2 * base_rows,
            2 * base_rows + 1,
            2 * cell,
            2 * cell + 1,
        ])
        cols = bm.concat([
            neighbour_cols,
            neighbour_cols,
            cell,
            cell,
        ])
        values = bm.concat([
            wx.reshape(-1),
            wy.reshape(-1),
            -bm.sum(wx, axis=1),
            -bm.sum(wy, axis=1),
        ])
        self._cell_lsq_gradient_matrix_cache = COOTensor(
            bm.stack([rows, cols], axis=0),
            values,
            spshape=(2 * NC, NC),
        ).coalesce().tocsr()
        return self._cell_lsq_gradient_matrix_cache

    def face_velocity(self, velocity, ap, pressure):
        """
        Compute Rhie-Chow corrected face velocity from cell fields.

        Parameters
        ----------
        velocity
            Cell-centered velocity with shape ``(NC, 2)`` or flattened layout
            ``[u, v]`` with length ``2*NC``.
        ap
            Momentum diagonal coefficients.
        pressure
            Cell-centered pressure.
        """
        NC = self.mesh.number_of_cells()
        edge_to_cell = self.fvm_geometry.face_to_cell
        if len(velocity.shape) == 1:
            velocity = bm.stack([velocity[:NC], velocity[NC:2 * NC]], axis=1)

        vf = 0.5 * (velocity[edge_to_cell[:, 0]] + velocity[edge_to_cell[:, 1]])
        is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
        if not bm.any(is_internal):
            return vf

        owner, neighbour, is_internal, d_pf, Sf, beta = (
            self._internal_pressure_flux_geometry(ap)
        )

        grad_p = self.gradient_reconstruct.cell_gradient(pressure)
        interp_grad = reconstruct_face_gradient(self.mesh, grad_p)[is_internal]
        jump = pressure[neighbour] - pressure[owner]
        d_dot_grad = bm.einsum("ij,ij->i", d_pf, interp_grad)
        correction_flux = -beta * (jump - d_dot_grad)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        normal_correction = (correction_flux / Sf_dot_Sf)[:, None] * Sf
        vf = bm.set_at(vf, is_internal, vf[is_internal] + normal_correction)
        return vf

    def boundary_velocity_rhs(self, boundary_velocity):
        """
        Assemble RHS contribution from known boundary face velocity.

        The continuity equation keeps internal face fluxes on the left-hand
        side. Known boundary flux ``u_b · S_b`` is moved to the right-hand side
        with a minus sign.
        """
        NC = self.mesh.number_of_cells()
        bd_face = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        owner = self.fvm_geometry.owner[bd_face]
        Sf = self.fvm_geometry.S_f[bd_face]
        flux = bm.einsum("ij,ij->i", boundary_velocity, Sf)
        rhs = bm.zeros(NC, dtype=flux.dtype)
        rhs = bm.index_add(rhs, owner, flux, axis=0, alpha=-self.rho)
        return rhs

    def _internal_pressure_flux_geometry(self, ap):
        """Return geometry for the internal pressure-response face flux.

        The pressure response vector is ``Q_f = D_f S_f``.  On non-orthogonal
        faces it is split as ``Q_f = beta_f d_f + C_f`` with
        ``beta_f = (Q_f dot S_f)/(d_f dot S_f)``.  The compact RC matrix uses
        ``beta_f (p_N - p_P)`` and the explicit part uses
        ``beta_f d_f dot grad(p)_f`` so all RC paths share one face-flux
        definition.
        """
        is_internal = self.fvm_geometry.is_internal
        owner = self.fvm_geometry.owner[is_internal]
        neighbour = self.fvm_geometry.neighbour[is_internal]
        D = self._cell_momentum_response(ap)
        Sf = self.fvm_geometry.S_f[is_internal]
        d_pf = self.fvm_geometry.d_f[is_internal]
        Df = 0.5 * (D[owner] + D[neighbour])
        response = Df * Sf
        numerator = bm.einsum("ij,ij->i", response, Sf)
        denominator = bm.einsum("ij,ij->i", Sf, d_pf)
        beta = numerator / denominator
        return owner, neighbour, is_internal, d_pf, Sf, beta

    def _cell_momentum_response(self, ap):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell_measure = mesh.entity_measure("cell")
        if ap.shape[0] == 2 * NC:
            ap_u = ap[:NC]
            ap_v = ap[NC:2 * NC]
        elif ap.shape[0] == NC:
            ap_u = ap
            ap_v = ap
        else:
            raise ValueError("ap must have length NC or 2*NC.")
        return bm.stack([cell_measure / ap_u, cell_measure / ap_v], axis=1)
