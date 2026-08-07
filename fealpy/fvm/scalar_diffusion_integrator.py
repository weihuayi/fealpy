"""Scalar finite-volume diffusion integrator for the orthogonal flux part."""

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.functionspace.utils import to_tensor_dof

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache
from fealpy.sparse import CSRTensor

from .fvm_geometry import FVMGeometry


def _local_dof_count(space: _FS) -> int:
    scalar_space = getattr(space, "scalar_space", None)
    if scalar_space is None:
        return int(space.number_of_local_dofs())
    return int(space.dof_numel * scalar_space.number_of_local_dofs())


class ScalarDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble the implicit two-point diffusion contribution.

    The local face matrix corresponds to the orthogonal finite-volume flux

        gamma_f |E_f| / |e_f| (phi_N - phi_P),

    where ``E_f`` is the projection of the face area vector onto the
    owner-neighbour centre line.  Non-orthogonal cross terms are intentionally
    not assembled here; they are handled explicitly by
    ``ScalarCrossDiffusionIntegrator``.
    """

    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 geometry: Optional[FVMGeometry]=None,
                 batched: bool=False,
                 method: str="over_relaxed",
                 nonorthogonal_eps: float=0.05) -> None:
        super().__init__()
        self.coef = coef
        self.q = 2 if q is None else q
        self.index = index
        self.geometry = geometry
        self.batched = batched
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = float(nonorthogonal_eps)
        if method not in self.assembly:
            raise ValueError(f"unknown diffusion method: {method!r}")
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        mesh = getattr(space, "mesh", None)
        geometry = self.geometry if self.geometry is not None else FVMGeometry(mesh, index=self.index)
        face_to_cell = geometry.face_to_cell

        scalar_space = getattr(space, "scalar_space", None)
        if scalar_space is None:
            return face_to_cell

        return to_tensor_dof(
            face_to_cell,
            space.dof_numel,
            scalar_space.number_of_global_dofs(),
            space.dof_priority,
        )

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
        geometry = self.geometry if self.geometry is not None else FVMGeometry(mesh, index=index)
        return geometry

    @variantmethod("over_relaxed")
    def assembly(self, space: _FS) -> TensorLike:
        geometry = self.fetch(space)
        decomposition = geometry.diffusion_face_decomposition("over_relaxed")
        return scalar_diffusion_local_matrix(
            space,
            _local_dof_count(space),
            coef=self.coef,
            orthogonal_factor=decomposition.orthogonal_factor,
        )

    @assembly.register("bounded_over_relaxed")
    def assembly(self, space: _FS) -> TensorLike:
        geometry = self.fetch(space)
        decomposition = geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=self.nonorthogonal_eps
        )
        return scalar_diffusion_local_matrix(
            space,
            _local_dof_count(space),
            coef=self.coef,
            orthogonal_factor=decomposition.orthogonal_factor,
        )

    @assembly.register("uncorrected")
    def assembly(self, space: _FS) -> TensorLike:
        return self.assembly["over_relaxed"](space)

def scalar_diffusion_local_matrix(
    space: _FS,
    local_dofs: int,
    *,
    coef: Optional[CoefLike]=None,
    orthogonal_factor: TensorLike,
) -> TensorLike:
    """Return local two-point matrices for the orthogonal diffusion flux.

    ``coef`` is interpreted as a constant or face-wise diffusion coefficient.
    Cell-wise coefficients must be interpolated to faces before calling this
    function.

    ``orthogonal_factor`` is supplied by ``DiffusionFaceDecomposition`` so the
    implicit matrix uses exactly the same face split as the explicit and
    boundary diffusion terms.
    """
    D = int(local_dofs)
    if coef is None:
        face_coef = bm.ones_like(orthogonal_factor, dtype=space.ftype)
    elif isinstance(coef, (int, float)):
        face_coef = bm.full_like(
            orthogonal_factor, fill_value=coef, dtype=space.ftype
        )
    else:
        face_coef = bm.array(
            coef,
            dtype=space.ftype,
            device=bm.get_device(orthogonal_factor),
        )
        if face_coef.shape == ():
            face_coef = (
                bm.ones_like(orthogonal_factor, dtype=space.ftype) * face_coef
            )
        elif (
            face_coef.ndim != 1
            or face_coef.shape[0] != orthogonal_factor.shape[0]
        ):
            raise ValueError(
                "coef must be scalar or face-wise for ScalarDiffusionIntegrator."
            )

    face_strength = orthogonal_factor * face_coef
    direction_matrix = bm.array(
        [[1.0, -1.0], [-1.0, 1.0]],
        dtype=space.ftype,
        device=bm.get_device(orthogonal_factor),
    )
    eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
    base_matrix = bm.einsum("ij,pq->ipjq", eye_D, direction_matrix).reshape(
        2 * D, 2 * D
    )
    return bm.einsum("i,ab->iab", face_strength, base_matrix)


class ScalarDiffusionMatrixAssembler:
    r"""Assemble the scalar diffusion matrix with a fixed CSR graph.

    This is the performance path corresponding to
    ``BilinearForm + ScalarDiffusionIntegrator`` for the orthogonal face flux

    .. math::

        \gamma_f \frac{|E_f|}{|d_f|}(\phi_P - \phi_N).

    Boundary conditions and pressure gauge constraints are not handled here.
    The sparsity pattern depends only on mesh topology; each assembly call
    updates only the values induced by the current face coefficient.
    """

    def __init__(
        self,
        space: _FS,
        *,
        geometry: Optional[FVMGeometry] = None,
        method: str = "over_relaxed",
        nonorthogonal_eps: float = 0.05,
    ) -> None:
        self.space = space
        self.mesh = getattr(space, "mesh", None)
        self.geometry = geometry if geometry is not None else FVMGeometry(self.mesh)
        self.NC = self.geometry.NC
        self.sparse_shape = (self.NC, self.NC)
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = float(nonorthogonal_eps)
        if method not in self.diffusion_face_factor:
            raise ValueError(f"unknown diffusion method: {method!r}")
        self.diffusion_face_factor.set(method)

        internal = bm.nonzero(self.geometry.is_internal)[0]
        owner = self.geometry.owner[internal]
        neighbour = self.geometry.neighbour[internal]
        self.internal = internal
        self.face_factor = self.diffusion_face_factor()

        rows = bm.concatenate([owner, owner, neighbour, neighbour])
        cols = bm.concatenate([owner, neighbour, owner, neighbour])
        self.face_template = bm.concatenate(
            [
                bm.ones_like(self.face_factor),
                -bm.ones_like(self.face_factor),
                -bm.ones_like(self.face_factor),
                bm.ones_like(self.face_factor),
            ]
        )
        internal_index = bm.arange(
            internal.shape[0],
            dtype=internal.dtype,
            device=bm.get_device(internal),
        )
        self.face_index = bm.concatenate(
            [internal_index, internal_index, internal_index, internal_index]
        )

        nrow, ncol = self.sparse_shape
        flat = bm.astype(rows, bm.int64) * ncol + bm.astype(cols, bm.int64)
        order = bm.argsort(flat)
        flat_sorted = flat[order]
        group_start = bm.ones(
            (flat.shape[0],),
            dtype=bm.bool,
            device=bm.get_device(flat),
        )
        group_start = bm.set_at(
            group_start,
            slice(1, None),
            flat_sorted[1:] != flat_sorted[:-1],
        )
        unique_flat = flat_sorted[group_start]
        group_id_sorted = bm.cumsum(group_start, axis=0) - 1
        self.entry_to_value = group_id_sorted[bm.argsort(order)]

        row = unique_flat // ncol
        col = unique_flat % ncol
        counts = bm.bincount(row, minlength=nrow)
        counts = bm.astype(counts, owner.dtype)
        self.crow = bm.concatenate(
            [
                bm.zeros((1,), dtype=owner.dtype, device=bm.get_device(owner)),
                bm.cumsum(counts, axis=0),
            ],
            axis=0,
        )
        self.col = bm.astype(col, owner.dtype)

    @variantmethod("over_relaxed")
    def diffusion_face_factor(self) -> TensorLike:
        decomposition = self.geometry.diffusion_face_decomposition("over_relaxed")
        return decomposition.orthogonal_factor[self.internal]

    @diffusion_face_factor.register("bounded_over_relaxed")
    def diffusion_face_factor(self) -> TensorLike:
        decomposition = self.geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=self.nonorthogonal_eps
        )
        return decomposition.orthogonal_factor[self.internal]

    @diffusion_face_factor.register("uncorrected")
    def diffusion_face_factor(self) -> TensorLike:
        return self.diffusion_face_factor["over_relaxed"]()

    def assembly(self, coef: TensorLike) -> CSRTensor:
        """Return the scalar diffusion matrix for the current face coefficient."""
        if isinstance(coef, (int, float)):
            coef_f = bm.full_like(self.face_factor, fill_value=coef)
        else:
            coef = bm.array(
                coef,
                dtype=self.face_factor.dtype,
                device=bm.get_device(self.face_factor),
            )
            if coef.shape == ():
                coef_f = bm.full_like(self.face_factor, fill_value=coef)
            else:
                coef_f = coef[self.internal]

        strength = coef_f * self.face_factor
        local_values = self.face_template * strength[self.face_index]
        values = bm.zeros(
            (self.col.shape[0],),
            dtype=local_values.dtype,
            device=bm.get_device(local_values),
        )
        values = bm.index_add(values, self.entry_to_value, local_values, axis=0)
        return CSRTensor(self.crow, self.col, values, spshape=self.sparse_shape)
