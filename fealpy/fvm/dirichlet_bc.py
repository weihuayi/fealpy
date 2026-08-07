"""Finite-volume Dirichlet boundary-condition algebra."""

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor, spdiags
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


class DirichletBC:
    """Apply prescribed boundary values to FVM matrices and RHS vectors.

    The class contains term-specific helpers because a Dirichlet value enters a
    finite-volume diffusion operator, convection boundary flux, and divergence
    block in different algebraic forms.  It only applies already-defined PDE
    boundary data to assembled algebraic systems; engineering boundary mapping
    and SIMPLE/PISO iteration rules live outside this class.
    """

    def __init__(
        self,
        geometry: FVMGeometry,
        faces: TensorLike,
        values: TensorLike,
        *,
        diffusion_method: str = "over_relaxed",
        nonorthogonal_eps: float = 0.05,
    ) -> None:
        """Store one explicit fixed-mesh Dirichlet patch."""
        if not isinstance(geometry, FVMGeometry):
            raise TypeError("geometry must be an FVMGeometry.")
        self.geometry = geometry
        self.faces = faces
        self.values = values
        if self.faces.ndim != 1:
            raise ValueError("faces must have shape (N,).")
        if self.values.shape[0] != self.faces.shape[0]:
            raise ValueError("values must have one entry per face.")
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = float(nonorthogonal_eps)
        if diffusion_method not in {
            "over_relaxed",
            "bounded_over_relaxed",
            "uncorrected",
        }:
            raise ValueError(
                f"unknown diffusion method: {diffusion_method!r}"
            )
        self.diffusion_method = diffusion_method

    def boundary_values(self, components: int) -> TensorLike:
        """Return boundary values compatible with the algebraic system shape."""
        value = self.values

        if components == 1:
            if value.ndim == 1:
                return value
            if value.ndim == 2 and value.shape[1] == 1:
                return value[:, 0]
            raise ValueError(
                "scalar Dirichlet system received vector boundary data; "
                "pass component=<index> or a scalar boundary function."
            )

        if value.ndim != 2 or value.shape[1] != components:
            raise ValueError(
                "vector Dirichlet system expects boundary data with shape "
                f"(N, {components}), got {value.shape}."
            )
        return value

    def diffusion_boundary_data(
        self,
        coef: float | TensorLike = 1.0,
    ) -> tuple[TensorLike, TensorLike]:
        """Return owner cells, implicit coefficients, and points for Dirichlet faces."""
        geometry = self.geometry
        decomposition = geometry.diffusion_face_decomposition(
            self.diffusion_method,
            eps=self.nonorthogonal_eps,
        )
        boundary_faces = self.faces
        boundary_integrator = decomposition.orthogonal_factor[
            boundary_faces
        ]
        if not isinstance(coef, (int, float)):
            if coef.shape == ():
                pass
            elif coef.ndim == 1 and coef.shape[0] == geometry.NF:
                coef = coef[boundary_faces]
            else:
                raise ValueError("coef must be scalar or face-wise.")
        boundary_integrator = coef * boundary_integrator
        return geometry.owner[boundary_faces], boundary_integrator

    def apply_diffusion_matrix(
        self,
        A: CSRTensor,
        coef: float | TensorLike = 1.0,
        *,
        components: int,
    ) -> CSRTensor:
        """Add the implicit Dirichlet diffusion diagonal to ``A``."""
        NC = self.geometry.NC
        if components < 1:
            raise ValueError("components must be positive.")
        expected = components * NC
        if A.shape != (expected, expected):
            raise ValueError(
                "matrix shape must match the explicit component count."
            )
        boundary_owner, boundary_integrator = self.diffusion_boundary_data(
            coef=coef,
        )
        boundary_diagonal = bm.zeros(
            NC,
            dtype=boundary_integrator.dtype,
            device=bm.get_device(boundary_integrator),
        )
        boundary_diagonal = bm.index_add(
            boundary_diagonal,
            boundary_owner,
            boundary_integrator,
            axis=0,
        )
        if components > 1:
            boundary_diagonal = bm.tile(boundary_diagonal, (components,))
        return A + spdiags(
            boundary_diagonal,
            0,
            A.shape[0],
            A.shape[1],
            index_dtype=A.itype,
        )

    def apply_diffusion_rhs(
        self,
        b: TensorLike,
        coef: float | TensorLike = 1.0,
        *,
        components: int,
    ) -> TensorLike:
        """Add the explicit Dirichlet diffusion RHS contribution to ``b``."""
        NC = self.geometry.NC
        if components < 1 or b.shape != (components * NC,):
            raise ValueError(
                "RHS shape must match the explicit component count."
            )
        boundary_owner, boundary_integrator = self.diffusion_boundary_data(
            coef=coef,
        )
        boundary_value = self.boundary_values(components)
        if components == 1:
            boundary_rhs = boundary_integrator * boundary_value
            return bm.index_add(b, boundary_owner, boundary_rhs, axis=0)

        boundary_rhs = boundary_integrator[:, None] * boundary_value
        boundary_rhs = bm.swapaxes(boundary_rhs, 0, 1).flatten()
        indices = bm.concat(
            [boundary_owner + component * NC for component in range(components)]
        )
        return bm.index_add(b, indices, boundary_rhs, axis=0)

    def apply_diffusion(
        self,
        A: CSRTensor,
        b: TensorLike,
        coef: float | TensorLike = 1.0,
        *,
        components: int,
    ) -> tuple[CSRTensor, TensorLike]:
        """Add boundary-face Dirichlet contribution for diffusion operators.

        For a boundary face, the prescribed value contributes an implicit
        owner-cell diagonal term and a matching RHS term.  This is the standard
        FVM face-flux form for Dirichlet data, and it supports scalar and
        component-wise vector fields.

        Args:
            A (sparse matrix): System matrix to be modified.
            b (ndarray): Right-hand side vector to be modified.

        Returns:
            tuple: (A, b)
                - A (sparse matrix): Modified system matrix with boundary conditions applied.
                - b (ndarray): Modified right-hand side vector with boundary contributions.
        """
        return (
            self.apply_diffusion_matrix(
                A,
                coef=coef,
                components=components,
            ),
            self.apply_diffusion_rhs(
                b,
                coef=coef,
                components=components,
            ),
        )

    def apply_convection(
        self,
        b: TensorLike,
        coef: TensorLike,
        *,
        components: int,
    ) -> TensorLike:
        """
        Apply Dirichlet boundary values to a finite-volume convection RHS.

        The interior convection operator only assembles owner-neighbour face
        contributions. On boundary faces the prescribed value contributes the
        known flux ``-(coef_f · S_f) g_D`` to the owner cell RHS.
        """
        geometry = self.geometry
        boundary_faces = self.faces
        NC = geometry.NC
        if components < 1 or b.shape != (components * NC,):
            raise ValueError(
                "RHS shape must match the explicit component count."
            )
        Sf = geometry.S_f[boundary_faces]
        if coef.ndim == 1 and coef.shape[0] == geometry.NF:
            flux = coef[boundary_faces]
        elif coef.ndim == 2 and coef.shape[0] == geometry.NF:
            flux = bm.einsum("ij,ij->i", coef[boundary_faces], Sf)
        else:
            raise ValueError("coef must be a face-wise scalar flux or vector face field.")

        boundary_owner = geometry.owner[boundary_faces]
        boundary_value = self.boundary_values(components)

        if components == 1:
            boundary_rhs = flux * boundary_value
            b = bm.index_add(b, boundary_owner, boundary_rhs, axis=0, alpha=-1)
            return b

        boundary_rhs = -flux[:, None] * boundary_value
        indices = bm.concat(
            [boundary_owner + component * NC for component in range(components)]
        )
        b = bm.index_add(b, indices, bm.swapaxes(boundary_rhs, 0, 1).flatten(), axis=0)
        return b
