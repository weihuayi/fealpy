"""Owner-oriented finite-volume face geometry."""

from dataclasses import dataclass
from fealpy.backend import backend_manager as bm
from fealpy.typing import Index, TensorLike, _S


def _prefix_sums(sizes):
    total = 0
    for size in sizes:
        total += int(size)
        yield total


@dataclass(frozen=True)
class DiffusionFaceDecomposition:
    """Geometry shared by all terms of one non-orthogonal diffusion scheme.

    ``E_f`` is parallel to the owner-neighbour vector and supplies the
    implicit two-point contribution.  ``T_f = S_f - E_f`` supplies the
    explicit cross-diffusion contribution.  ``orthogonal_factor`` is the
    geometry-only coefficient ``|E_f| / |d_f|``.
    """

    E_f: TensorLike
    mag_E_f: TensorLike
    T_f: TensorLike
    orthogonal_factor: TensorLike


def boundary_face_flag(points, selector):
    """Evaluate a boundary selector on Cartesian points.

    ``points`` has shape ``(..., GD)`` and the selector returns one boolean
    value for every point, with shape ``(...)``.
    """
    if not callable(selector):
        raise ValueError("selector must be callable.")

    flag = bm.array(
        selector(points),
        dtype=bm.bool,
        device=bm.get_device(points),
    )
    if flag.shape == points.shape[:-1]:
        return flag
    raise ValueError(
        "selector must return a boolean array with shape points.shape[:-1]."
    )


def select_boundary_faces(geometry, selector):
    """Return global boundary-face indices selected at face centers."""
    boundary_faces = geometry.boundary_faces
    flag = boundary_face_flag(
        geometry.face_center[boundary_faces],
        selector,
    )
    return boundary_faces[flag]


class FVMGeometry:
    """Collect basic owner-oriented face geometry for FVM operators.

    The class intentionally stores only mesh geometry.  It does not define PDE
    coefficients, boundary values, gradient reconstruction, Rhie-Chow response,
    or pressure-correction data.
    """

    def __init__(self, mesh, *, index: Index = _S) -> None:
        self.mesh = mesh
        self.index = index
        self._diffusion_decomposition_cache = {}
        self._linear_owner_weight_cache = None

        if not hasattr(mesh, "Entities"):
            raise RuntimeError(
                "FVMGeometry requires the Mesh/EntityView/Relation interface."
            )

        cell_views = mesh.Entities(-1)
        face_views = mesh.Entities(-2)
        if not cell_views:
            raise ValueError("FVMGeometry requires at least one cell sector.")
        if not face_views:
            raise ValueError("FVMGeometry requires at least one face sector.")

        cell_sizes = [view.size() for view in cell_views]
        face_sizes = [view.size() for view in face_views]
        self.cell_sector_offsets = tuple(
            [0] + list(_prefix_sums(cell_sizes))
        )
        self.face_sector_offsets = tuple(
            [0] + list(_prefix_sums(face_sizes))
        )
        self.NC = self.cell_sector_offsets[-1]
        total_faces = self.face_sector_offsets[-1]
        self.cell_views = tuple(cell_views)
        self.face_views = tuple(face_views)
        self.cell_sector_slices = tuple(
            slice(start, end)
            for start, end in zip(
                self.cell_sector_offsets[:-1], self.cell_sector_offsets[1:]
            )
        )
        self.face_sector_slices = tuple(
            slice(start, end)
            for start, end in zip(
                self.face_sector_offsets[:-1], self.face_sector_offsets[1:]
            )
        )

        self.cell_center = bm.concatenate(
            [view.barycenter() for view in cell_views], axis=0
        )
        self.GD = self.cell_center.shape[1]
        self.cell_measure = bm.concatenate(
            [view.measure() for view in cell_views], axis=0
        )

        face_centers = []
        face_measures = []
        face_normals = []
        for view in face_views:
            center = view.barycenter()
            measure = view.measure()
            normal = view.normal()
            if normal.ndim == 3:
                if normal.shape[1] != 1:
                    raise ValueError(
                        "FVMGeometry requires one geometric normal per face."
                    )
                normal = normal[:, 0, :]
            if normal.ndim != 2:
                raise ValueError(
                    "face normal must have shape (NF, GD) or (NF, 1, GD)."
                )
            normal_norm = bm.linalg.norm(normal, axis=1)
            if bool(bm.to_numpy(bm.any(normal_norm <= 0.0))):
                raise ValueError("face normal has zero length.")

            face_centers.append(center)
            face_measures.append(measure)
            face_normals.append(normal / normal_norm[:, None])

        full_face_center = bm.concatenate(face_centers, axis=0)
        full_face_measure = bm.concatenate(face_measures, axis=0)
        full_unit_normal = bm.concatenate(face_normals, axis=0)

        incidence_face = []
        incidence_cell = []
        incidence_local_face = []
        cell_face_counts = []
        for cell_id, cell_view in enumerate(cell_views):
            cell_offset = self.cell_sector_offsets[cell_id]
            local_face_offset = 0
            for face_id, face_view in enumerate(face_views):
                try:
                    relation = cell_view.to(face_view)
                except KeyError:
                    continue
                if relation.src_indices is not None or relation.tgt_indices.ndim != 2:
                    raise ValueError(
                        "FVMGeometry currently requires fixed-width cell-to-face relations."
                    )

                local_faces = relation.tgt_indices
                cell_count, local_face_count = local_faces.shape
                if cell_count != cell_sizes[cell_id]:
                    raise ValueError("cell-to-face relation has an invalid cell count.")

                incidence_face.append(
                    bm.reshape(local_faces, (-1,))
                    + self.face_sector_offsets[face_id]
                )
                local_cells = bm.arange(
                    cell_count,
                    dtype=local_faces.dtype,
                    device=bm.get_device(local_faces),
                )
                incidence_cell.append(
                    bm.repeat(local_cells + cell_offset, local_face_count)
                )
                incidence_local_face.append(
                    bm.tile(
                        bm.arange(
                            local_face_count,
                            dtype=local_faces.dtype,
                            device=bm.get_device(local_faces),
                        ) + local_face_offset,
                        cell_count,
                    )
                )
                local_face_offset += local_face_count
            cell_face_counts.append(
                bm.full(
                    (cell_sizes[cell_id],),
                    local_face_offset,
                    dtype=cell_view.indices.dtype,
                    device=bm.get_device(cell_view.indices),
                )
            )

        if not incidence_face:
            raise ValueError("mesh has no cell-to-face incidences.")

        incidence_face = bm.concatenate(incidence_face, axis=0)
        incidence_cell = bm.concatenate(incidence_cell, axis=0)
        incidence_local_face = bm.concatenate(incidence_local_face, axis=0)
        incidence_count = bm.bincount(incidence_face, minlength=total_faces)
        invalid_incidence = (incidence_count < 1) | (incidence_count > 2)
        if bool(bm.to_numpy(bm.any(invalid_incidence))):
            raise ValueError(
                "FVMGeometry requires a manifold mesh with one or two cells per face."
            )

        order = bm.argsort(incidence_face)
        sorted_cell = incidence_cell[order]
        sorted_local_face = incidence_local_face[order]
        group_end = bm.cumsum(incidence_count, axis=0)
        group_start = group_end - incidence_count
        first = group_start
        last = group_end - 1

        full_owner = sorted_cell[first]
        full_neighbour = sorted_cell[last]
        full_owner_local = sorted_local_face[first]
        full_neighbour_local = sorted_local_face[last]

        face_to_cell = bm.stack([full_owner, full_neighbour], axis=1)[index]
        owner_local_face = full_owner_local[index]
        neighbour_local_face = full_neighbour_local[index]
        if face_to_cell.ndim == 1:
            face_to_cell = face_to_cell[None, :]
            owner_local_face = owner_local_face[None]
            neighbour_local_face = neighbour_local_face[None]

        self.face_to_cell = face_to_cell
        self.owner_local_face = owner_local_face
        self.neighbour_local_face = neighbour_local_face
        self.owner = self.face_to_cell[:, 0]
        self.neighbour = self.face_to_cell[:, 1]
        self.is_internal = self.owner != self.neighbour
        self.is_boundary = ~self.is_internal
        self.boundary_faces = bm.nonzero(self.is_boundary)[0]

        self.face_center = full_face_center[index]
        self.face_measure = full_face_measure[index]
        if self.face_center.ndim == 1:
            self.face_center = self.face_center[None, :]
            self.face_measure = self.face_measure[None]
        self.NF = self.face_center.shape[0]
        self.cell_face_count = bm.concatenate(cell_face_counts, axis=0)

        owner_to_neighbour = self.cell_center[self.neighbour] - self.cell_center[self.owner]
        owner_to_face = self.face_center - self.cell_center[self.owner]
        self.d_f = bm.where(self.is_internal[:, None], owner_to_neighbour, owner_to_face)
        self.mag_d_f = bm.linalg.norm(self.d_f, axis=1)
        if bool(bm.to_numpy(bm.any(self.mag_d_f <= 0.0))):
            raise ValueError("face centre vector has zero length.")

        S_f = self.face_measure[:, None] * full_unit_normal[index]
        if S_f.ndim == 1:
            S_f = S_f[None, :]

        projection = bm.einsum("ij,ij->i", S_f, self.d_f)
        S_f = bm.where(projection[:, None] < 0.0, -S_f, S_f)
        self.S_f = S_f
        self.mag_S_f = bm.linalg.norm(self.S_f, axis=1)
        if bool(bm.to_numpy(bm.any(self.mag_S_f <= 0.0))):
            raise ValueError("face area vector has zero length.")
        self.n_f = self.S_f / self.mag_S_f[:, None]

        owner_projection = bm.einsum("ij,ij->i", self.S_f, self.d_f)
        if bool(bm.to_numpy(bm.any(owner_projection <= 0.0))):
            raise ValueError("face area vector is not owner-oriented.")

        self.boundary_owner_to_face_vector = owner_to_face[self.is_boundary]
        boundary_normal = self.n_f[self.is_boundary]
        self.boundary_normal_distance = bm.einsum("ij,ij->i", self.boundary_owner_to_face_vector, boundary_normal)
        if bool(bm.to_numpy(bm.any(self.boundary_normal_distance <= 0.0))):
            raise ValueError("boundary face has zero owner-normal distance.")

    def cell_integral(self, integrand, *, q: int = 3):
        """Integrate one Cartesian function over all cell sectors.

        ``integrand(points, cell_slice)`` receives physical quadrature points
        for one homogeneous cell sector and the corresponding slice in the
        global FVM cell ordering.  Sector results are concatenated in that
        ordering.
        """
        values = []
        for view, cell_slice in zip(self.cell_views, self.cell_sector_slices):
            values.append(
                view.integral(
                    lambda points, cell_slice=cell_slice: integrand(
                        points, cell_slice
                    ),
                    q=q,
                )
            )
        return bm.concatenate(values, axis=0)

    def face_integral(self, integrand, *, q: int = 3):
        """Integrate one Cartesian function over all selected face sectors."""
        values = []
        for view, face_slice in zip(self.face_views, self.face_sector_slices):
            values.append(
                view.integral(
                    lambda points, face_slice=face_slice: integrand(
                        points, face_slice
                    ),
                    q=q,
                )
            )
        return bm.concatenate(values, axis=0)[self.index]

    def diffusion_face_decomposition(
        self,
        method: str = "over_relaxed",
        *,
        eps: float = 0.05,
    ) -> DiffusionFaceDecomposition:
        """Return the cached face decomposition for one diffusion variant.

        The returned object is the single geometry source for the implicit
        two-point term, explicit cross-diffusion term, and boundary closure.
        Field-dependent cross-flux limiting is deliberately not a geometry
        decomposition and is configured by the cross-diffusion operator.
        """
        supported = {
            "over_relaxed",
            "bounded_over_relaxed",
            "uncorrected",
        }
        if method not in supported:
            raise ValueError(f"unknown diffusion method: {method!r}")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        key = (method, float(eps))
        cached = self._diffusion_decomposition_cache.get(key)
        if cached is not None:
            return cached

        if method == "bounded_over_relaxed":
            projection = bm.einsum("ij,ij->i", self.n_f, self.d_f)
            denominator = bm.maximum(projection, eps * self.mag_d_f)
            E_f = (self.mag_S_f / denominator)[:, None] * self.d_f
        else:
            S_dot_S = bm.einsum("ij,ij->i", self.S_f, self.S_f)
            d_dot_S = bm.einsum("ij,ij->i", self.d_f, self.S_f)
            if bm.any(d_dot_S <= 0.0):
                raise ValueError(
                    "over-relaxed decomposition has invalid d_f dot S_f."
                )
            E_f = (S_dot_S / d_dot_S)[:, None] * self.d_f

        mag_E_f = bm.linalg.norm(E_f, axis=1)
        T_f = self.S_f - E_f
        decomposition = DiffusionFaceDecomposition(
            E_f=E_f,
            mag_E_f=mag_E_f,
            T_f=T_f,
            orthogonal_factor=mag_E_f / self.mag_d_f,
        )
        self._diffusion_decomposition_cache[key] = decomposition
        return decomposition

    def normal_distance(self, faces: Index = _S):
        """Return the projection of ``d_f`` onto the owner-oriented unit normal."""
        d_f = self.d_f[faces]
        n_f = self.n_f[faces]
        if d_f.ndim == 1:
            return bm.einsum("i,i->", d_f, n_f)
        return bm.einsum("ij,ij->i", d_f, n_f)

    def linear_owner_weight(self):
        """Return owner-side linear interpolation weights on faces.

        Internal faces use owner/neighbour distances projected onto the face
        area vector direction.  Boundary faces return one because no real
        neighbour cell participates in the interpolation.
        """
        weight = self._linear_owner_weight_cache
        if weight is not None:
            return weight
        owner_dist = bm.abs(
            bm.einsum(
                "ij,ij->i",
                self.S_f,
                self.face_center - self.cell_center[self.owner],
            )
        )
        neighbour_dist = bm.abs(
            bm.einsum(
                "ij,ij->i",
                self.S_f,
                self.cell_center[self.neighbour] - self.face_center,
            )
        )
        total_dist = owner_dist + neighbour_dist
        weight = bm.where(total_dist > 0.0, neighbour_dist / total_dist, 0.5)
        weight = bm.where(self.is_internal, weight, 1.0)
        self._linear_owner_weight_cache = weight
        return weight

    def scatter_face_flux_to_cells(self, face_flux):
        """Scatter owner-oriented face fluxes to cell flux sums.

        ``face_flux[f]`` must be oriented with this geometry's ``S_f``.  For an
        internal face between owner ``P`` and neighbour ``N``, a positive
        flux means flow in the ``P -> N`` direction, so the cell flux sums are

            cell_sum[P] += face_flux[f],
            cell_sum[N] -= face_flux[f].

        For a boundary face, only the owner cell receives the face contribution.
        The returned value is the finite-volume boundary flux sum on each cell;
        it is not divided by cell volume and should not be interpreted as a
        cell-average divergence.
        """
        if face_flux.ndim == 0:
            raise ValueError("face_flux must have at least one dimension.")
        if face_flux.shape[0] != self.owner.shape[0]:
            raise ValueError(
                f"face_flux has {face_flux.shape[0]} faces, expected "
                f"{self.owner.shape[0]}."
            )

        result = bm.zeros(
            (self.NC,) + tuple(face_flux.shape[1:]),
            dtype=face_flux.dtype,
            device=bm.get_device(face_flux),
        )
        result = bm.index_add(result, self.owner, face_flux, axis=0)
        return bm.index_add(
            result,
            self.neighbour[self.is_internal],
            face_flux[self.is_internal],
            axis=0,
            alpha=-1,
        )


def face_interpolation_owner_weight(
    geometry: FVMGeometry,
    *,
    method: str = "linear",
) -> TensorLike:
    """Return owner-side face interpolation weights.

    ``linear`` is the geometry-consistent face interpolation used by the
    collocated FVM operators.  ``average`` is the arithmetic central weight.
    Boundary faces return one because no neighbour cell participates.
    """
    if method not in {"average", "linear"}:
        raise ValueError("method must be 'average' or 'linear'.")

    if method == "linear":
        return geometry.linear_owner_weight()

    weight = 0.5 * bm.ones_like(geometry.mag_S_f)
    return bm.where(geometry.is_internal, weight, 1.0)


def interpolate_cell_to_face(
    cell_values: TensorLike,
    *,
    geometry: FVMGeometry,
    method: str = "linear",
) -> TensorLike:
    """Interpolate cell values to faces with the shared geometric weights.

    The first axis of ``cell_values`` is the control-volume axis.  Additional
    axes are preserved, so the same operation applies to scalar, vector, and
    tensor fields.  Boundary faces use their owner value.
    """
    if cell_values.ndim == 0:
        raise ValueError("cell_values must have a control-volume axis.")
    if cell_values.shape[0] != geometry.NC:
        raise ValueError(
            f"cell_values has {cell_values.shape[0]} cells, expected "
            f"{geometry.NC}."
        )

    owner_weight = face_interpolation_owner_weight(
        geometry,
        method=method,
    )
    weight_shape = (owner_weight.shape[0],) + (1,) * (cell_values.ndim - 1)
    owner_weight = bm.reshape(owner_weight, weight_shape)
    return (
        owner_weight * cell_values[geometry.owner]
        + (1.0 - owner_weight) * cell_values[geometry.neighbour]
    )
