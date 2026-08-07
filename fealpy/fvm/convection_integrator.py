"""Finite-volume convection integrator for face-velocity fluxes."""

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator.variantmethod import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.functionspace.utils import to_tensor_dof

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache
from fealpy.sparse import CSRTensor

from .fvm_geometry import FVMGeometry, face_interpolation_owner_weight


class ConvectionIntegrator(LinearInt, OpInt, FaceInt):
    r"""Assemble the central finite-volume convection operator.

    ``coef`` is a face-wise convection velocity.  In incompressible momentum
    equations it may already include the density factor, so the face flux used
    by this integrator is always interpreted as

    .. math::

        \phi_f = \mathbf c_f \cdot \mathbf S_f .

    The ``interpolation`` option only selects the owner/neighbour weights used
    to reconstruct the central face value.  It does not switch to an upwind,
    bounded, or limited convection scheme.

    Boundary flux closure is deliberately outside this low-level operator.
    Dirichlet, Neumann, and natural outlet convection contributions are applied
    by the boundary-condition layer or by the flow solver that owns the case
    semantics.
    """

    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 interpolation: str="average",
                 index: Index=_S,
                 geometry: Optional[FVMGeometry]=None,
                 batched: bool=False) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.interpolation = self._validate_interpolation(interpolation)
        self.index = index
        self.geometry = geometry
        self.batched = batched

    @staticmethod
    def _validate_interpolation(interpolation: str) -> str:
        if interpolation not in {"average", "linear"}:
            raise ValueError("interpolation must be 'average' or 'linear'.")
        return interpolation

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

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Assemble the central face stencil without basis evaluation.

        The current FVM convection operator is P0 cell-centred, so the local
        face matrix depends only on the face flux, owner/neighbour weights, and
        the number of tensor components.
        """
        coef = self.coef
        mesh = getattr(space, "mesh", None)
        geometry = self.geometry if self.geometry is not None else FVMGeometry(mesh, index=self.index)
        Sf = geometry.S_f
        D = getattr(space, "dof_numel", 1)
        eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
        owner_weight = face_interpolation_owner_weight(
            geometry,
            method=self.interpolation,
        )
        neighbour_weight = 1.0 - owner_weight
        direction_matrix = bm.stack(
            [
                bm.stack([owner_weight, neighbour_weight], axis=-1),
                bm.stack([-owner_weight, -neighbour_weight], axis=-1),
            ],
            axis=1,
        )
        base_matrix = bm.einsum("ij,fpq->fipjq", eye_D, direction_matrix).reshape(-1, 2 * D, 2 * D)
        if coef is None:
            coef = bm.stack([bm.ones_like(Sf[:, 0]), bm.zeros_like(Sf[:, 0])], axis=1)
        integrator = bm.einsum("ij,ij->i", Sf, coef)
        return integrator[:, None, None] * base_matrix


class ConvectionMatrixAssembler:
    r"""Assemble the implicit two-point convection matrix with fixed CSR graph.

    This is the performance path for the same central face operator represented
    by ``ConvectionIntegrator``:

    .. math::

        \phi_f
        \begin{bmatrix}
        w_P & w_N\\
        -w_P & -w_N
        \end{bmatrix}.

    The sparsity pattern depends only on mesh topology, component layout, and
    interpolation weights.  Each call to :meth:`assembly` updates only the CSR
    values induced by the current face velocity.
    """

    def __init__(
        self,
        space: _FS,
        *,
        interpolation: str = "average",
        geometry: Optional[FVMGeometry] = None,
    ) -> None:
        self.space = space
        self.mesh = getattr(space, "mesh", None)
        self.geometry = geometry if geometry is not None else FVMGeometry(self.mesh)
        self.interpolation = ConvectionIntegrator._validate_interpolation(interpolation)
        self.S_f = self.geometry.S_f
        self.GD = getattr(space, "dof_numel", 1)
        self.NF = self.S_f.shape[0]
        self.sparse_shape = (space.number_of_global_dofs(),) * 2

        face_to_cell = self.geometry.face_to_cell
        scalar_space = getattr(space, "scalar_space", None)
        if scalar_space is None:
            face_to_dof = face_to_cell
        else:
            face_to_dof = to_tensor_dof(
                face_to_cell,
                space.dof_numel,
                scalar_space.number_of_global_dofs(),
                space.dof_priority,
            )

        owner_weight = face_interpolation_owner_weight(
            self.geometry,
            method=self.interpolation,
        )
        neighbour_weight = 1.0 - owner_weight
        direction_matrix = bm.stack(
            [
                bm.stack([owner_weight, neighbour_weight], axis=-1),
                bm.stack([-owner_weight, -neighbour_weight], axis=-1),
            ],
            axis=1,
        )
        eye = bm.eye(self.GD, dtype=space.ftype, device=bm.get_device(space))
        local_template = bm.einsum("ij,fpq->fipjq", eye, direction_matrix).reshape(
            self.NF,
            2 * self.GD,
            2 * self.GD,
        )

        local_shape = local_template.shape
        rows = bm.broadcast_to(face_to_dof[:, :, None], local_shape).reshape(-1)
        cols = bm.broadcast_to(face_to_dof[:, None, :], local_shape).reshape(-1)
        template = local_template.reshape(-1)
        face_index = bm.repeat(
            bm.arange(self.NF, dtype=face_to_dof.dtype, device=bm.get_device(face_to_dof)),
            (2 * self.GD) * (2 * self.GD),
        )

        active = bm.nonzero(template != 0.0)[0]
        rows = rows[active]
        cols = cols[active]
        self.template = template[active]
        self.face_index = face_index[active]

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
        counts = bm.astype(counts, face_to_dof.dtype)
        self.crow = bm.concatenate(
            [
                bm.zeros((1,), dtype=face_to_dof.dtype, device=bm.get_device(face_to_dof)),
                bm.cumsum(counts, axis=0),
            ],
            axis=0,
        )
        self.col = bm.astype(col, face_to_dof.dtype)

    def assembly(self, coef: TensorLike) -> CSRTensor:
        """Return the convection matrix for the current face velocity."""
        if coef is None:
            components = [bm.ones_like(self.S_f[:, 0])]
            components += [
                bm.zeros_like(self.S_f[:, 0])
                for _ in range(self.S_f.shape[1] - 1)
            ]
            coef = bm.stack(components, axis=1)

        flux = bm.einsum("ij,ij->i", self.S_f, coef)
        local_values = self.template * flux[self.face_index]
        values = bm.zeros(
            (self.col.shape[0],),
            dtype=local_values.dtype,
            device=bm.get_device(local_values),
        )
        values = bm.index_add(values, self.entry_to_value, local_values, axis=0)
        return CSRTensor(self.crow, self.col, values, spshape=self.sparse_shape)
