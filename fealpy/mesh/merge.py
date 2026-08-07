from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial import KDTree

from ..backend import bm
from ..backend import Tensor
from .storage import EntitySector, MeshBlock, Relation
from .view import Mesh

__all__ = ["merge_positions_by_distance", "merge_mesh_storage", "merge"]


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], rank: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def merge_positions_by_distance(
    position: Tensor,
    /,
    *,
    tol: float,
    leafsize: int = 16,
) -> tuple[Tensor, Tensor]:
    """Merge points whose Euclidean distance is within ``tol``.

    Parameters:
        position:
            Point array with shape ``(n_points, n_dim)``.
        tol:
            Distance tolerance used for deduplication. Must be non-negative.
        leafsize:
            Leaf size used to build ``scipy.spatial.KDTree``.

    Returns:
        A tuple ``(new_position, old_to_new)`` where:
        - ``new_position`` is the deduplicated point array.
        - ``old_to_new`` maps each original point index to index in ``new_position``.
    """
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol!r}")

    if leafsize <= 0:
        raise ValueError(f"leafsize must be positive, got {leafsize!r}")

    pos_np = np.asarray(position)

    if pos_np.ndim != 2:
        raise ValueError(
            f"position must be a rank-2 tensor shaped (n_points, n_dim), got shape {pos_np.shape!r}"
        )

    n_points = len(pos_np)
    if n_points == 0:
        empty_map = bm.asarray([], dtype=bm.int64)
        return bm.asarray(pos_np), empty_map

    if n_points == 1:
        singleton_map = bm.asarray([0], dtype=bm.int64)
        return bm.asarray(pos_np), singleton_map

    tree = KDTree(pos_np, leafsize=leafsize)
    pairs: Iterable[tuple[int, int]] = tree.query_pairs(r=tol)

    parent = list(range(n_points))
    rank = [0] * n_points
    for i, j in pairs:
        _union(parent, rank, i, j)

    roots = [_find(parent, i) for i in range(n_points)]

    comp_min_index: dict[int, int] = {}
    for idx, root in enumerate(roots):
        if root not in comp_min_index or idx < comp_min_index[root]:
            comp_min_index[root] = idx

    # Keep deterministic ordering by first appearance in original position.
    component_roots = sorted(comp_min_index.keys(), key=lambda r: comp_min_index[r])
    representative_indices = [comp_min_index[r] for r in component_roots]

    root_to_new = {root: new_idx for new_idx, root in enumerate(component_roots)}
    old_to_new_np = np.asarray([root_to_new[root] for root in roots], dtype=np.int64)

    new_position = bm.asarray(pos_np[representative_indices])
    old_to_new = bm.asarray(old_to_new_np, dtype=bm.int64)
    return new_position, old_to_new


def _stable_unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2D, got shape {rows.shape!r}")

    unique_sorted, first_idx, inverse_sorted = np.unique(
        np.sort(rows, axis=1),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    del unique_sorted

    # numpy.unique sorts rows; reorder by first appearance to keep stable output.
    stable_to_sorted = np.argsort(first_idx)
    stable_first_idx = first_idx[stable_to_sorted]

    sorted_to_stable = np.empty_like(stable_to_sorted)
    sorted_to_stable[stable_to_sorted] = np.arange(len(stable_to_sorted))

    stable_inverse = np.empty_like(inverse_sorted, dtype=np.int64)
    stable_inverse[:] = sorted_to_stable[inverse_sorted]

    unique_rows = rows[stable_first_idx]
    return unique_rows, stable_inverse, stable_first_idx


def _remap_block_with_position(
    block: EntitySector,
    pos_old_to_new: np.ndarray,
) -> tuple[EntitySector, Tensor, np.ndarray]:
    old_indices = np.asarray(block.indices)
    if old_indices.ndim != 2:
        raise ValueError(
            f"block {block.schema_name!r} indices must be rank-2, got shape {old_indices.shape!r}"
        )

    remapped_indices = pos_old_to_new[old_indices]
    new_indices_np, old_to_new_np, representative_old_idx = _stable_unique_rows(remapped_indices)

    new_sector = EntitySector(
        schema_name=block.schema_name,
        indices=bm.asarray(new_indices_np),
        attributes=dict(block.attributes),
    )
    return new_sector, bm.asarray(old_to_new_np, dtype=bm.int64), representative_old_idx


def _remap_relation(
    relation: Relation,
    entity_maps_np: dict[str, np.ndarray],
    entity_representatives_np: dict[str, np.ndarray],
) -> Relation:
    src_name = relation.src_name
    tgt_name = relation.tgt_name

    src_old_to_new = entity_maps_np[src_name]
    tgt_old_to_new = entity_maps_np[tgt_name]

    tgt_indices_old = np.asarray(relation.tgt_indices)
    tgt_indices_new = tgt_old_to_new[tgt_indices_old]

    if relation.src_indices is None:
        representative_old_src = entity_representatives_np[src_name]

        tgt_indices_out_np = tgt_indices_new[representative_old_src]
        return Relation(
            src_name=src_name,
            tgt_name=tgt_name,
            tgt_indices=bm.asarray(tgt_indices_out_np),
            src_indices=None,
        )

    src_indices_old = np.asarray(relation.src_indices)
    src_indices_new = src_old_to_new[src_indices_old]

    key_cols = [np.reshape(src_indices_new, (-1, 1)), np.reshape(tgt_indices_new, (len(tgt_indices_new), -1))]
    key_rows = np.concatenate(key_cols, axis=1)
    _, keep = np.unique(key_rows, axis=0, return_index=True)
    keep.sort()

    return Relation(
        src_name=src_name,
        tgt_name=tgt_name,
        tgt_indices=bm.asarray(tgt_indices_new[keep]),
        src_indices=bm.asarray(src_indices_new[keep], dtype=bm.int64),
    )


def merge_mesh_storage(
    storage: MeshBlock,
    /,
    *,
    tol: float,
    leafsize: int = 16,
) -> tuple[MeshBlock, dict[str, Tensor]]:
    """Merge one mesh storage by distance-based point deduplication.

    Returns:
        A tuple ``(new_storage, entity_old_to_new)`` where:
        - ``new_storage`` is the merged storage.
        - ``entity_old_to_new`` maps each schema name to old-entity -> new-entity indices.
    """
    new_positions, pos_old_to_new = merge_positions_by_distance(
        storage.positions,
        tol=tol,
        leafsize=leafsize,
    )
    pos_old_to_new_np = np.asarray(pos_old_to_new)

    new_sectors: dict[str, EntitySector] = {}
    entity_old_to_new: dict[str, Tensor] = {}
    entity_maps_np: dict[str, np.ndarray] = {}
    entity_representatives_np: dict[str, np.ndarray] = {}

    for schema_name, block in storage.sectors.items():
        new_sector, old_to_new, representative_old_idx = _remap_block_with_position(block, pos_old_to_new_np)
        new_sectors[schema_name] = new_sector
        entity_old_to_new[schema_name] = old_to_new
        entity_maps_np[schema_name] = np.asarray(old_to_new)
        entity_representatives_np[schema_name] = representative_old_idx

    new_relations: dict[tuple[str, str], Relation] = {}
    for key, relation in storage.relations.items():
        new_relations[key] = _remap_relation(relation, entity_maps_np, entity_representatives_np)

    new_storage = MeshBlock(
        positions=new_positions,
        sectors=new_sectors,
        relations=new_relations,
        root_entity_names=list(storage.root_entity_names),
    )

    return new_storage, entity_old_to_new


def merge(
    mesh: Mesh,
    /,
    *,
    tol: float = 1e-5,
    leafsize: int = 16,
) -> tuple[Mesh, dict[str, Tensor]]:
    """Merge a mesh by deduplicating close points and remapping entities/relations.

    Parameters:
        mesh (Mesh): The input mesh to merge.
        tol (float): Distance tolerance used for deduplication. Must be non-negative.
        leafsize (int): Leaf size used to build ``scipy.spatial.KDTree``.
            In a KDTree, leaf_size specifies the maximum number of data points
            stored in a leaf node. When building the tree, the algorithm
            recursively splits the dataset until each leaf contains at most
            leaf_size points. If a region has fewer points than leaf_size,
            it becomes a leaf without further splitting.
            Smaller leaf_size values create deeper trees because the data is
            split more frequently, while larger values produce shallower trees
            with fewer levels.

    Returns:
        A tuple (new_mesh, entity_old_to_new) where:
        - ``new_mesh`` is the merged mesh.
        - ``entity_old_to_new`` maps each schema name to old-entity -> new-entity indices.
    """
    storage, entity_old_to_new = merge_mesh_storage(mesh.block, tol=tol, leafsize=leafsize)
    return Mesh(storage), entity_old_to_new
