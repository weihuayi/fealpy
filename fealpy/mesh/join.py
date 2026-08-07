from __future__ import annotations

from ..backend import bm
from .storage import EntitySector, MeshBlock, Relation
from .view import Mesh

__all__ = ["join", "join_mesh_storage"]


def _merge_root_names(left: MeshBlock, right: MeshBlock) -> list[str]:
    roots = list(left.root_entity_names)
    for name in right.root_entity_names:
        if name not in roots:
            roots.append(name)
    return roots


def _merge_block(
    schema_name: str,
    left_block: EntitySector | None,
    right_block: EntitySector | None,
    node_offset: int,
) -> EntitySector:
    if left_block is None and right_block is None:
        raise ValueError(f"missing blocks for schema {schema_name!r}")

    arrays = []
    attributes: dict = {}

    if left_block is not None:
        arrays.append(left_block.indices)
        attributes.update(left_block.attributes)

    if right_block is not None:
        arrays.append(right_block.indices + node_offset)
        attributes.update(right_block.attributes)

    indices = arrays[0] if len(arrays) == 1 else bm.concat(arrays, axis=0)
    return EntitySector(schema_name=schema_name, indices=indices, attributes=attributes)


def _merge_relation(
    key: tuple[str, str],
    left_rel: Relation | None,
    right_rel: Relation | None,
    src_offset: int,
    tgt_offset: int,
) -> Relation:
    src_name, tgt_name = key

    if left_rel is None and right_rel is None:
        raise ValueError(f"missing relation for key {key!r}")

    if right_rel is not None:
        tgt_indices_right = right_rel.tgt_indices + tgt_offset
        src_indices_right = None if right_rel.src_indices is None else right_rel.src_indices + src_offset

    if left_rel is None:
        assert right_rel is not None
        return Relation(
            src_name=src_name,
            tgt_name=tgt_name,
            tgt_indices=tgt_indices_right,
            src_indices=src_indices_right,
        )

    if right_rel is None:
        return Relation(
            src_name=src_name,
            tgt_name=tgt_name,
            tgt_indices=left_rel.tgt_indices,
            src_indices=left_rel.src_indices,
        )

    tgt_indices = bm.concat([left_rel.tgt_indices, tgt_indices_right], axis=0)

    if left_rel.src_indices is None and src_indices_right is None:
        src_indices = None
    elif left_rel.src_indices is not None and src_indices_right is not None:
        src_indices = bm.concat([left_rel.src_indices, src_indices_right], axis=0)
    else:
        raise ValueError(
            f"incompatible relation src_indices for key {key!r}: one side is None and the other is not"
        )

    return Relation(
        src_name=src_name,
        tgt_name=tgt_name,
        tgt_indices=tgt_indices,
        src_indices=src_indices,
    )


def join_mesh_storage(left: MeshBlock, right: MeshBlock) -> MeshBlock:
    """Join two mesh storages by pure concatenation."""
    node_offset = len(left.positions)
    positions = bm.concat([left.positions, right.positions], axis=0)

    sectors: dict[str, EntitySector] = {}
    sector_names = list(left.sectors)
    for name in right.sectors:
        if name not in sectors and name not in sector_names:
            sector_names.append(name)

    for name in sector_names:
        sectors[name] = _merge_block(
            schema_name=name,
            left_block=left.sectors.get(name),
            right_block=right.sectors.get(name),
            node_offset=node_offset,
        )

    src_offsets: dict[str, int] = {
        name: len(sec.indices)
        for name, sec in left.sectors.items()
    }

    relations: dict[tuple[str, str], Relation] = {}
    relation_keys = list(left.relations)
    for key in right.relations:
        if key not in relations and key not in relation_keys:
            relation_keys.append(key)

    for key in relation_keys:
        src_name, tgt_name = key
        relations[key] = _merge_relation(
            key=key,
            left_rel=left.relations.get(key),
            right_rel=right.relations.get(key),
            src_offset=src_offsets.get(src_name, 0),
            tgt_offset=src_offsets.get(tgt_name, 0),
        )

    return MeshBlock(
        positions=positions,
        sectors=sectors,
        relations=relations,
        root_entity_names=_merge_root_names(left, right),
    )


def join(left: Mesh, right: Mesh, /) -> Mesh:
    """Join two meshes by pure concatenation.

    Returns:
        Mesh: The joined mesh.

    Notes:
        - This function only concatenates positions and remaps indices and relations.
        - It does NOT deduplicate points/entities or perform distance-based merge.
    """
    return Mesh(join_mesh_storage(left.block, right.block))
