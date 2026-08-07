
from collections.abc import Iterable
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from ..backend import bm
from ..backend import Tensor, dtype

if TYPE_CHECKING:
    from .schema import EntitySchema
    from .view.mesh_view import MeshView


__all__ = [
    "MultiIndex",
    "multi_index_sort",
    "multi_index_tensorprod",
    "to_ipoint",
    "to_ipoint_permutation",
    "ipoints",
]


class MultiIndex:
    @classmethod
    def multi_index_matrix(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index matrix for interpolation points of
        degree p with n vertices. The multi-index matrix is of shape
        (C(p+n-1, n-1), n) and each row corresponds to the multi-index of
        an interpolation point.

        Parameters:
            p (int): Degree of interpolation.
            n (int): Number of vertices in the Simplex.
            dtype (dtype, optional): Data type of the output tensor. If None, it will
                default to int32.

        Returns:
            Tensor: A tensor of shape (C(p+n-1, n-1), n) containing the multi-indices.
        """
        if dtype is None:
            dtype = bm.int32

        sep = bm.flip(bm.asarray(
            tuple(combinations_with_replacement(range(p+1), n-1)),
            dtype=dtype
        ), axis=0)
        raw = bm.zeros((sep.shape[0], n+1), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @classmethod
    def multi_index_inner(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index corresponding to the inner interpolation
        points of degree p with n vertices.

        See also: `multi_index_matrix`."""
        if p < n:
            if dtype is None:
                dtype = bm.int32
            return bm.zeros((0, n), dtype=dtype)
        return cls.multi_index_matrix(p - n, n, dtype=dtype) + 1


def multi_index_sort(multi_index: Tensor, /) -> Tensor:
    """Return the indices to sort multi-indices according to the predefined orientation."""
    NV = multi_index.shape[-1]
    count = bm.sum(multi_index != 0, axis=1)
    nonzero_row, nonzero_col = bm.nonzero(multi_index)
    rank = bm.zeros_like(count, dtype=bm.int64)
    rank = bm.index_add(rank, nonzero_row, NV**nonzero_col) # type: ignore[call-overload]
    arg = bm.lexsort(tuple(multi_index.T) + (rank, count)) # type: ignore[call-overload]
    return arg


def multi_index_sort_by_locals(multi_index: Tensor, local_faces: dict[str, list[list[int]]], /) -> Tensor:
    num_multi_index = multi_index.shape[0]
    NV = multi_index.shape[-1]
    TOTAL = bm.sum(multi_index, axis=-1)
    topdim = bm.zeros((num_multi_index,), dtype=bm.int8)
    face_type_index = bm.zeros((num_multi_index,), dtype=bm.int8)
    face_instance_index = bm.zeros((num_multi_index,), dtype=bm.int8)
    weights = bm.zeros_like(multi_index, dtype=bm.int64)

    from .schema.registry import SCHEMA_REGISTRY

    for fti, (key, value) in enumerate(local_faces.items()):
        schema_cls = SCHEMA_REGISTRY[key]
        TD = schema_cls.top_dim

        for fii, local_face in enumerate(value):
            mask = bm.logical_and(
                bm.sum(multi_index[:, local_face], axis=-1) == TOTAL,
                bm.all(multi_index[:, local_face] != 0, axis=-1)
            )
            mask_idx = bm.nonzero(mask)[0]
            topdim[mask_idx] = TD
            face_type_index[mask_idx] = fti
            face_instance_index[mask_idx] = fii
            new_weights = bm.zeros((mask_idx.shape[0], NV), dtype=bm.int64)
            new_weights[:, local_face] = NV**bm.arange(len(local_face))[None, :]
            weights[mask_idx] = new_weights

    mask = bm.all(multi_index != 0, axis=-1)
    topdim[mask] = 127

    rank = bm.sum(multi_index * weights, axis=-1)
    arg = bm.lexsort((rank, face_instance_index, face_type_index, topdim))
    return arg


def multi_index_tensorprod(
    broadcast_multi_index: Tensor,
    split_indices: tuple[int, ...] | None = None
) -> Tensor:
    """Compute the tensor product between split multi-indices.

    Do nothing if split_indices is None, as no other operand is provided to
    perform the tensor product with."""
    from functools import reduce

    if split_indices is not None:
        mi_tuple = bm.split(broadcast_multi_index, split_indices, axis=-1)
    else:
        return broadcast_multi_index

    def kron_last_dim(a: Tensor, b: Tensor) -> Tensor:
        if bm.size(a) == 0:
            return a
        return (a[..., :, None] * b[..., None, :]).reshape(*a.shape[:-1], -1) # type: ignore[return-value]

    return reduce(kron_last_dim, reversed(mi_tuple))


def to_ipoint(mesh: "MeshView", name: str, order: int) -> Tensor: # [num_entities, num_ip]
    """Get the interpolation point indices for the given entity and order,
    in unstructured meshes.
    The interpolation point indices are ordered from lower-dimensional
    sub-entities to higher-dimensional entities, and the interpolation points
    of each sub-entity are ordered according to the vertex orientation.

    Parameters:
        mesh (Mesh): The mesh object.
        name (str): The name of the entity shape (e.g., "tri", "tet", "edge").
        order (int): The degree of interpolation.

    Returns:
        Tensor: A tensor of shape (num_entities, num_ip) containing the
            interpolation point indices.
    """
    collected = []
    dim_cursor = 0
    ip_cursor = 0
    tgt_entity = mesh.Entity(name)
    shutdown = False

    while True:
        for subentity in mesh.Entities(dim_cursor):
            ### (1) Get ip mapping from sub-entity to the global
            num_sub_entity = subentity.size()
            num_internal_ip = subentity.num_multi_index(order, internal=True)
            if num_internal_ip == 0:
                dim_cursor += 1
                continue
            sub_map = bm.arange(
                ip_cursor,
                ip_cursor + num_sub_entity * num_internal_ip,
                dtype=bm.int64,
            )
            sub_map = bm.reshape(sub_map, (num_sub_entity, num_internal_ip))

            if dim_cursor == tgt_entity.schema.top_dim:
                full_map = bm.reshape(sub_map, (tgt_entity.size(), -1))
                collected.append(full_map)
                shutdown = True
                break

            ### (2) Permute the ip according to the vertex orientation
            tgt_to_sub = tgt_entity.to(subentity).tgt_indices # [num_entities, num_local_subs]
            full_map = bm.reshape(sub_map[tgt_to_sub], (-1, num_internal_ip))
            # [num_entities * num_local_subs, num_internal_ip]
            global_vo = tgt_entity.global_permutations(subentity.schema.name) # [num_entities, num_local_subs, num_subs_vertex]
            global_vo = bm.reshape(global_vo, (-1, global_vo.shape[-1]))
            # [num_entities * num_local_subs, num_subs_vertex]

            for vo, do in subentity.schema.vo_to_do((order,)).items():
                vo = bm.asarray(vo, dtype=bm.uint8, device=full_map.device)
                vo_mask = bm.all(global_vo == vo[None, :], axis=-1) # [num_entities * num_local_subs]
                full_map = bm.where(vo_mask[:, None], full_map[:, do], full_map)

            full_map = bm.reshape(full_map, (tgt_entity.size(), -1))
            # [num_entities, num_local_subs * num_internal_ip]

            ### (3) Collect the ip mapping
            collected.append(full_map)
            ip_cursor += num_sub_entity * num_internal_ip
            dim_cursor += 1
        else:
            if dim_cursor > tgt_entity.schema.top_dim:
                break

        if shutdown:
            break

    result = bm.concat(collected, axis=1)
    permutation = to_ipoint_permutation(tgt_entity.schema, (order,))

    if permutation is not None:
        permutation = bm.device_put(permutation, result.device)
        result = result[:, permutation]
    return result


def to_ipoint_permutation(schema: type["EntitySchema"], order: tuple[int, ...]) -> Tensor | None:
    """Column permutation from topological ipoint order to basis order.

    ``to_ipoint`` builds interpolation-point indices in topological order
    (lower-dimensional sub-entities first). Schemas whose basis functions
    use a different local ordering may override this hook to return the
    column permutation that aligns the mapping with ``multi_index`` and
    shape-function order.
    """
    mi = schema.multi_index(order, tensorprod=True)
    natural_to_topological = multi_index_sort_by_locals(mi, schema.OFace)
    return bm.argsort(natural_to_topological)


def ipoints(mesh: "MeshView", order: int | tuple[int, ...], names: Iterable[str]) -> Tensor:
    """Get the interpolation points for the given entity and order.

    Parameters:
        mesh (MeshView): The mesh object.
        order (int | tuple[int, ...]): The degree of interpolation.
        names (Iterable[str]): The names of the entities for which to compute
            interpolation points. For example, ["tet", "hex"].

    Returns:
        Tensor: A tensor of shape (num_ip, GD) containing the interpolation points.
    """
    if isinstance(order, int):
        order = (order,)
    if not order or any(p <= 0 for p in order):
        raise ValueError(f"order must be positive, got {order!r}")

    device = mesh.block.positions.device

    collected = []
    for subentity in [mesh.Entity(entity) for entity in names]:
        mi = subentity.schema.multi_index(order, internal=True)
        mi = bm.device_put(mi, device)
        if mi.shape[0] == 0:
            continue

        vertices = subentity.indices
        points = mesh.block.positions[vertices]
        weights = mi / bm.sum(mi, axis=-1, keepdims=True)
        points = bm.einsum("qv, evd -> eqd", weights, points)
        collected.append(bm.reshape(points, (-1, mesh.geo_dimension())))

    if not collected:
        return bm.zeros((0, mesh.geo_dimension()), dtype=mesh.block.positions.dtype, device=device)

    return bm.concat(collected, axis=0)
