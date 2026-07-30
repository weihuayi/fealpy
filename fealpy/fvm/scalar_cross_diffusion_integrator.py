from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .fvm_geometry import FVMGeometry


_BOUNDARY_POLICIES = {"all", "zero"}
_CROSS_FLUX_LIMITERS = {"none", "orthogonal_flux_ratio"}


class ScalarCrossDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble explicit non-orthogonal diffusion correction as a cell RHS.

    At solver level this term is part of a non-orthogonal correction loop.  At
    operator level it is the cross-diffusion source induced by the tangential
    correction vector in the face-vector decomposition.

    The implicit matrix part is handled by ``ScalarDiffusionIntegrator``.  The
    same ``method`` name must be passed to both integrators so that ``E_f`` and
    ``T_f`` come from one face-vector decomposition.  This integrator scatters
    the face correction flux

        coef_f * C_f · grad(phi)_f,

    to owner/neighbour cells.  The default ``over_relaxed`` method uses the
    ordinary over-relaxed ``T_f``.  Boundary fluxes are retained by default;
    callers that require zero boundary correction must select
    ``boundary_policy="zero"`` explicitly.
    """

    def __init__(
        self,
        uh=None,
        grad_f=None,
        coef: Optional[CoefLike]=None,
        q: Optional[int]=None,
        *,
        face_flux_correction=None,
        correction_vector=None,
        geometry=None,
        boundary_policy: str="all",
        cross_flux_limiter: str="none",
        limit_coeff: float=0.5,
        limiter_small: float=1.0e-30,
        nonorthogonal_eps: float=0.05,
        index: Index=_S,
        batched: bool=False,
        method: str="over_relaxed",
    ) -> None:
        super().__init__()
        self.uh = uh
        self.grad_f = grad_f
        self.coef = coef
        self.face_flux_correction = face_flux_correction
        self.correction_vector = correction_vector
        self.geometry = geometry
        if boundary_policy not in _BOUNDARY_POLICIES:
            raise ValueError(f"Unsupported boundary_policy: {boundary_policy!r}")
        self.boundary_policy = boundary_policy
        if cross_flux_limiter not in _CROSS_FLUX_LIMITERS:
            raise ValueError(
                f"unknown cross_flux_limiter: {cross_flux_limiter!r}"
            )
        self.cross_flux_limiter = cross_flux_limiter
        self.limit_coeff = limit_coeff
        self.limiter_small = limiter_small
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = nonorthogonal_eps
        self.q = 2 if q is None else q
        self.index = index
        self.batched = batched
        if method not in self.assembly:
            raise ValueError(f"unknown diffusion method: {method!r}")
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, "mesh", None)
        geometry = self.geometry if self.geometry is not None else FVMGeometry(mesh, index=index)
        return geometry.face_to_cell, geometry

    @variantmethod("over_relaxed")
    def assembly(self, space: _FS) -> TensorLike:
        face_to_cell, geometry = self.fetch(space)
        decomposition = geometry.diffusion_face_decomposition(
            "over_relaxed"
        )
        return self._assemble_from_vector(
            space,
            face_to_cell,
            geometry,
            decomposition.T_f,
            decomposition.orthogonal_factor,
        )

    @assembly.register("bounded_over_relaxed")
    def assembly(self, space: _FS) -> TensorLike:
        face_to_cell, geometry = self.fetch(space)
        decomposition = geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=self.nonorthogonal_eps
        )
        return self._assemble_from_vector(
            space,
            face_to_cell,
            geometry,
            decomposition.T_f,
            decomposition.orthogonal_factor,
        )

    @assembly.register("uncorrected")
    def assembly(self, space: _FS) -> TensorLike:
        face_to_cell, geometry = self.fetch(space)
        decomposition = geometry.diffusion_face_decomposition("uncorrected")
        return self._assemble_from_vector(
            space,
            face_to_cell,
            geometry,
            bm.zeros_like(geometry.S_f),
            decomposition.orthogonal_factor,
        )

    def _assemble_from_vector(
        self,
        space: _FS,
        face_to_cell: TensorLike,
        geometry: FVMGeometry,
        correction_vector: TensorLike,
        orthogonal_factor: TensorLike,
    ) -> TensorLike:
        face_flux = scalar_cross_diffusion_face_flux(
            space,
            geometry,
            face_to_cell,
            grad_f=self.grad_f,
            coef=self.coef,
            face_flux_correction=self.face_flux_correction,
            correction_vector=(
                self.correction_vector
                if self.correction_vector is not None
                else correction_vector
            ),
            boundary_policy=self.boundary_policy,
        )
        if self.cross_flux_limiter == "orthogonal_flux_ratio":
            face_flux = limit_cross_diffusion_face_flux(
                face_flux,
                self.uh,
                geometry,
                orthogonal_factor=orthogonal_factor,
                coef=self.coef,
                limit_coeff=self.limit_coeff,
                limiter_small=self.limiter_small,
            )
        return geometry.scatter_face_flux_to_cells(face_flux)


def scalar_cross_diffusion_face_flux(
    space: _FS,
    geometry: FVMGeometry,
    face_to_cell: TensorLike,
    *,
    grad_f=None,
    coef: Optional[CoefLike]=None,
    face_flux_correction=None,
    correction_vector=None,
    boundary_policy: str="all",
) -> TensorLike:
    """Return owner-oriented flux for a preselected correction vector."""
    if boundary_policy not in _BOUNDARY_POLICIES:
        raise ValueError(f"Unsupported boundary_policy: {boundary_policy!r}")

    if face_flux_correction is not None:
        face_flux = bm.array(
            face_flux_correction,
            dtype=space.ftype,
            device=bm.get_device(geometry.cell_center),
        )
    else:
        if grad_f is None:
            raise ValueError("grad_f is required when face_flux_correction is not provided.")
        if correction_vector is None:
            raise ValueError(
                "correction_vector is required when face_flux_correction is not provided."
            )
        correction_vector = bm.array(
            correction_vector,
            dtype=space.ftype,
            device=bm.get_device(grad_f),
        )

        if grad_f.ndim == 2:
            face_flux = bm.einsum("ij,ij->i", correction_vector, grad_f)
        elif grad_f.ndim == 3:
            face_flux = bm.einsum("ij,ikj->ik", correction_vector, grad_f)
        else:
            raise ValueError(f"Unsupported grad_f shape: {grad_f.shape}")

        shape_source = face_flux[:, 0] if face_flux.ndim == 2 else face_flux
        if coef is None:
            face_coef = bm.ones_like(shape_source, dtype=space.ftype)
        elif isinstance(coef, (int, float)):
            face_coef = bm.full_like(shape_source, fill_value=coef, dtype=space.ftype)
        else:
            face_coef = bm.array(
                coef,
                dtype=space.ftype,
                device=bm.get_device(face_flux),
            )
        if face_flux.ndim == 1:
            face_flux = bm.einsum("i,i->i", face_coef, face_flux)
        else:
            face_flux = face_flux * face_coef[:, None]

    if boundary_policy == "all":
        return face_flux

    is_boundary = face_to_cell[:, 0] == face_to_cell[:, 1]
    if face_flux.ndim == 1:
        return bm.where(is_boundary, 0.0, face_flux)
    if face_flux.ndim == 2:
        return bm.where(is_boundary[:, None], 0.0, face_flux)
    raise ValueError(f"Unsupported face_flux_correction shape: {face_flux.shape}")


def limit_cross_diffusion_face_flux(
    face_flux: TensorLike,
    uh: TensorLike,
    geometry: FVMGeometry,
    *,
    orthogonal_factor: TensorLike,
    coef: Optional[CoefLike]=None,
    limit_coeff: float,
    limiter_small: float,
) -> TensorLike:
    """Limit scalar cross-diffusion relative to its implicit face flux."""
    if face_flux.ndim != 1:
        raise ValueError("cross-flux limiting currently supports scalar face flux only.")
    if uh is None:
        raise ValueError(
            "uh is required with cross_flux_limiter='orthogonal_flux_ratio'."
        )
    uh = bm.array(
        uh,
        dtype=face_flux.dtype,
        device=bm.get_device(face_flux),
    )
    if uh.ndim != 1:
        raise ValueError("cross-flux limiting currently supports scalar cell values only.")
    if not 0.0 <= limit_coeff <= 1.0:
        raise ValueError("limit_coeff must be in [0, 1].")
    if limiter_small <= 0.0:
        raise ValueError("limiter_small must be positive.")

    internal = geometry.is_internal
    owner = geometry.owner
    neighbour = geometry.neighbour
    orthogonal_factor = bm.array(
        orthogonal_factor,
        dtype=face_flux.dtype,
        device=bm.get_device(face_flux),
    )
    if orthogonal_factor.shape != face_flux.shape:
        raise ValueError("orthogonal_factor must have one value per face.")
    if coef is None:
        face_coef = bm.ones_like(face_flux)
    elif isinstance(coef, (int, float)):
        face_coef = bm.full_like(face_flux, fill_value=coef)
    else:
        face_coef = bm.array(
            coef,
            dtype=face_flux.dtype,
            device=bm.get_device(face_flux),
        )
        if face_coef.shape == ():
            face_coef = bm.full_like(face_flux, fill_value=face_coef)
        elif face_coef.shape != face_flux.shape:
            raise ValueError("coef must be scalar or have one value per face.")
    orthogonal_flux = bm.zeros_like(face_flux)
    cell_jump = bm.abs(uh[neighbour[internal]] - uh[owner[internal]])
    orthogonal_flux = bm.set_at(
        orthogonal_flux,
        internal,
        bm.abs(face_coef[internal]) * orthogonal_factor[internal] * cell_jump,
    )
    numerator = limit_coeff * orthogonal_flux[internal]
    denominator = (
        (1.0 - limit_coeff) * bm.abs(face_flux[internal]) + limiter_small
    )
    limiter = bm.ones_like(face_flux)
    limiter = bm.set_at(
        limiter,
        internal,
        bm.minimum(numerator / denominator, bm.ones_like(numerator)),
    )
    return face_flux * limiter


class CrossDiffusionRHSAssembler:
    r"""Assemble explicit non-orthogonal diffusion RHS by direct face scatter.

    The solver-level name is ``nonorthogonal`` correction.  This assembler keeps
    the lower-level ``cross_diffusion`` wording because it only builds the
    explicit tangential diffusion flux/RHS used by that correction.

    This is the high-frequency performance path equivalent to
    ``LinearForm + ScalarCrossDiffusionIntegrator``.  The mathematical object is
    a dense cell RHS,

    .. math::

        b_K = \sum_{f \in \partial K} \mu_f \nabla_f \phi \cdot T_f,

    so there is no sparse matrix graph to cache.  The cached state is the FVM
    face geometry and scatter layout supplied by ``FVMGeometry``.
    """

    def __init__(
        self,
        space: _FS,
        *,
        geometry: Optional[FVMGeometry]=None,
        method: str="over_relaxed",
        cross_flux_limiter: str="none",
        nonorthogonal_eps: float=0.05,
    ) -> None:
        self.space = space
        self.mesh = getattr(space, "mesh", None)
        self.geometry = geometry if geometry is not None else FVMGeometry(self.mesh)
        self.face_to_cell = self.geometry.face_to_cell
        self.is_tensor_space = getattr(space, "scalar_space", None) is not None
        self.dof_priority = getattr(space, "dof_priority", True)
        if cross_flux_limiter not in _CROSS_FLUX_LIMITERS:
            raise ValueError(
                f"unknown cross_flux_limiter: {cross_flux_limiter!r}"
            )
        self.cross_flux_limiter = cross_flux_limiter
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = float(nonorthogonal_eps)
        if method not in self.assembly:
            raise ValueError(f"unknown diffusion method: {method!r}")
        self.assembly.set(method)

    @variantmethod("over_relaxed")
    def assembly(self, **kwargs) -> TensorLike:
        decomposition = self.geometry.diffusion_face_decomposition(
            "over_relaxed"
        )
        return self._assembly_from_vector(
            decomposition.T_f,
            decomposition.orthogonal_factor,
            **kwargs,
        )

    @assembly.register("bounded_over_relaxed")
    def assembly(self, **kwargs) -> TensorLike:
        decomposition = self.geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=self.nonorthogonal_eps
        )
        return self._assembly_from_vector(
            decomposition.T_f,
            decomposition.orthogonal_factor,
            **kwargs,
        )

    @assembly.register("uncorrected")
    def assembly(self, **kwargs) -> TensorLike:
        decomposition = self.geometry.diffusion_face_decomposition("uncorrected")
        return self._assembly_from_vector(
            bm.zeros_like(self.geometry.S_f),
            decomposition.orthogonal_factor,
            **kwargs,
        )

    def _assembly_from_vector(
        self,
        correction_vector: TensorLike,
        orthogonal_factor: TensorLike,
        *,
        uh=None,
        grad_f=None,
        coef: Optional[CoefLike]=None,
        face_flux_correction=None,
        boundary_policy: str="all",
        limit_coeff: float=0.5,
        limiter_small: float=1.0e-30,
    ) -> TensorLike:
        """Return the explicit correction RHS for the current face gradients."""
        face_flux = scalar_cross_diffusion_face_flux(
            self.space,
            self.geometry,
            self.face_to_cell,
            grad_f=grad_f,
            coef=coef,
            face_flux_correction=face_flux_correction,
            correction_vector=correction_vector,
            boundary_policy=boundary_policy,
        )
        if self.cross_flux_limiter == "orthogonal_flux_ratio":
            face_flux = limit_cross_diffusion_face_flux(
                face_flux,
                uh,
                self.geometry,
                orthogonal_factor=orthogonal_factor,
                coef=coef,
                limit_coeff=limit_coeff,
                limiter_small=limiter_small,
            )
        rhs = self.geometry.scatter_face_flux_to_cells(face_flux)
        if not self.is_tensor_space or rhs.ndim == 1:
            return rhs
        if self.dof_priority:
            return bm.swapaxes(rhs, 0, 1).reshape(-1)
        return rhs.reshape(-1)
