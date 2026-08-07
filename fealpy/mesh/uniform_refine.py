from __future__ import annotations

from collections.abc import Callable

from ..backend import bm
from ..backend import Tensor
from ..sparse import csr_matrix
from .storage import MeshBlock
from .topology.builder import TopologyBuilder

__all__ = [
    "uniform_refine",
    "uniform_refine_segment",
    "uniform_refine_triangle",
    "uniform_refine_quadrilateral",
    "uniform_refine_tetrahedron",
    "uniform_refine_prism",
    "uniform_refine_hexahedron",
]


RefineFunc = Callable[..., list | None]


def _arange_like(start: int, stop: int, ref: Tensor) -> Tensor:
    return bm.arange(start, stop, dtype=ref.dtype, device=bm.get_device(ref))


def _empty_like(shape: tuple[int, int], ref: Tensor) -> Tensor:
    return bm.zeros(shape, dtype=ref.dtype, device=bm.get_device(ref))


def _barycenter(block: MeshBlock, indices: Tensor) -> Tensor:
    return bm.mean(block.positions[indices], axis=1)


def _root_names(block: MeshBlock) -> set[str]:
    return set(block.root_entity_names)


def _rebuild_topology(block: MeshBlock) -> None:
    roots = _root_names(block)
    block.sectors = {name: sector for name, sector in block.sectors.items() if name in roots}
    block.relations.clear()
    block._cache_boundary_info = None
    TopologyBuilder.construct(block)


def _ensure_topology(block: MeshBlock, *required: tuple[str, str]) -> None:
    missing_sector = any(name not in block.sectors for pair in required for name in pair)
    missing_relation = any(pair not in block.relations for pair in required)
    if missing_sector or missing_relation:
        _rebuild_topology(block)


def _sector(block: MeshBlock, name: str) -> Tensor:
    return block.get_sector(name).indices


def _set_sector(block: MeshBlock, name: str, indices: Tensor) -> None:
    block.get_sector(name).indices = indices


def _append_positions(block: MeshBlock, *positions: Tensor) -> None:
    block.positions = bm.concat([block.positions, *positions], axis=0)


def _prolongation_from_edges(
    old_positions: Tensor,
    edge: Tensor,
    edge2new_node: Tensor,
) -> csr_matrix:
    nn = len(old_positions)
    ne = len(edge)
    shape = (nn + ne, nn)
    values = bm.ones(nn + 2 * ne, **bm.context(old_positions))
    values = bm.set_at(values, bm.arange(nn, nn + 2 * ne), 0.5)

    i0 = bm.arange(nn, dtype=edge.dtype, device=bm.get_device(edge))
    i = bm.concat((i0, edge2new_node, edge2new_node), axis=0)
    j = bm.concat((i0, edge[:, 0], edge[:, 1]), axis=0)
    return csr_matrix((values, (i, j)), shape)


def _refine_segment_once(block: MeshBlock) -> csr_matrix | None:
    segment = _sector(block, "segment")
    node = block.positions
    nn = len(node)
    nc = len(segment)

    new_node = (node[segment[:, 0]] + node[segment[:, 1]]) / 2
    new_index = _arange_like(nn, nn + nc, segment)
    _append_positions(block, new_node)

    left = bm.stack((segment[:, 0], new_index), axis=1)
    right = bm.stack((new_index, segment[:, 1]), axis=1)
    _set_sector(block, "segment", bm.concat((left, right), axis=0))
    _rebuild_topology(block)


def uniform_refine_segment(block: MeshBlock, n: int = 1, returnim: bool = False, **kwargs) -> list | None:
    """Uniformly refine segment entities in a MeshBlock in place."""
    im = [] if returnim else None
    for _ in range(n):
        if returnim:
            segment = _sector(block, "segment")
            node = block.positions
            nc = len(segment)
            nn = len(node)
            new_index = _arange_like(nn, nn + nc, segment)
            im.append(_prolongation_from_edges(node, segment, new_index))
        _refine_segment_once(block)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_triangle_once(block: MeshBlock) -> csr_matrix | None:
    _ensure_topology(block, ("tri", "segment"))
    tri = _sector(block, "tri")
    edge = _sector(block, "segment")
    cell2edge = block.relations[("tri", "segment")].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)

    edge2new = _arange_like(nn, nn + ne, tri)
    new_node = (node[edge[:, 0]] + node[edge[:, 1]]) / 2
    _append_positions(block, new_node)

    p = bm.concat((tri, edge2new[cell2edge]), axis=1)
    new_tri = bm.concat(
        (p[:, [0, 5, 4]], p[:, [5, 1, 3]], p[:, [4, 3, 2]], p[:, [3, 4, 5]]),
        axis=0,
    )
    _set_sector(block, "tri", new_tri)
    _rebuild_topology(block)


def uniform_refine_triangle(
    block: MeshBlock,
    n: int = 1,
    surface=None,
    interface=None,
    returnim: bool = False,
    **kwargs,
) -> list | None:
    """Uniformly refine triangular entities in a MeshBlock in place."""
    im = [] if returnim else None
    for _ in range(n):
        _ensure_topology(block, ("tri", "segment"))
        if returnim:
            edge = _sector(block, "segment")
            node = block.positions
            edge2new = _arange_like(len(node), len(node) + len(edge), _sector(block, "tri"))
            im.append(_prolongation_from_edges(node, edge, edge2new))
        _refine_triangle_once(block)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_quadrilateral_once(block: MeshBlock) -> None:
    _ensure_topology(block, ("quad", "segment"))
    quad = _sector(block, "quad")
    edge = _sector(block, "segment")
    cell2edge = block.relations[("quad", "segment")].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nc = len(quad)

    edge_center = _barycenter(block, edge)
    cell_center = _barycenter(block, quad)
    edge2center = _arange_like(nn, nn + ne, quad)
    cell_center_idx = _arange_like(nn + ne, nn + ne + nc, quad)[:, None]

    cp = [quad[:, i:i + 1] for i in range(4)]
    ep = [edge2center[cell2edge[:, i]][:, None] for i in range(4)]
    new_quad = _empty_like((4 * nc, 4), quad)
    new_quad = bm.set_at(new_quad, (slice(0, None, 4), slice(None)), bm.concat([cp[0], ep[0], ep[3], cell_center_idx], axis=1))
    new_quad = bm.set_at(new_quad, (slice(1, None, 4), slice(None)), bm.concat([ep[0], cp[1], cell_center_idx, ep[1]], axis=1))
    new_quad = bm.set_at(new_quad, (slice(2, None, 4), slice(None)), bm.concat([cell_center_idx, ep[1], ep[2], cp[3]], axis=1))
    new_quad = bm.set_at(new_quad, (slice(3, None, 4), slice(None)), bm.concat([ep[3], cell_center_idx, cp[2], ep[2]], axis=1))

    _append_positions(block, edge_center, cell_center)
    _set_sector(block, "quad", new_quad)
    _rebuild_topology(block)


def uniform_refine_quadrilateral(
    block: MeshBlock,
    n: int = 1,
    surface=None,
    interface=None,
    returnim: bool = False,
    **kwargs,
) -> list | None:
    """Uniformly refine quadrilateral entities in a MeshBlock in place."""
    im = [] if returnim else None
    for _ in range(n):
        _ensure_topology(block, ("quad", "segment"))
        if returnim:
            quad = _sector(block, "quad")
            edge = _sector(block, "segment")
            node = block.positions
            nn = len(node)
            ne = len(edge)
            nc = len(quad)
            shape = (nn + ne + nc, nn)
            values = bm.ones(nn + 2 * ne + 4 * nc, **bm.context(node))
            values = bm.set_at(values, bm.arange(nn, nn + 2 * ne), 0.5)
            values = bm.set_at(values, bm.arange(nn + 2 * ne, nn + 2 * ne + 4 * nc), 0.25)
            i0 = bm.arange(nn, dtype=quad.dtype, device=bm.get_device(quad))
            i1 = _arange_like(nn, nn + ne, quad)
            i2 = _arange_like(nn + ne, nn + ne + nc, quad)
            i = bm.concat((i0, i1, i1, i2, i2, i2, i2), axis=0)
            j = bm.concat((i0, edge[:, 0], edge[:, 1], quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]), axis=0)
            im.append(csr_matrix((values, (i, j)), shape))
        _refine_quadrilateral_once(block)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_tetrahedron_once(block: MeshBlock) -> None:
    _ensure_topology(block, ("tet", "segment"))
    tet = _sector(block, "tet")
    edge = _sector(block, "segment")
    cell2edge = block.relations[("tet", "segment")].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nc = len(tet)

    edge2new = _arange_like(nn, nn + ne, tet)
    new_node = (node[edge[:, 0]] + node[edge[:, 1]]) / 2
    _append_positions(block, new_node)

    p = edge2new[cell2edge]
    new_tet = _empty_like((8 * nc, 4), tet)
    new_tet = bm.set_at(new_tet, (slice(4 * nc), 3), tet.T.flatten())
    new_tet = bm.set_at(new_tet, (slice(nc), slice(3)), p[:, [0, 2, 1]])
    new_tet = bm.set_at(new_tet, (slice(nc, 2 * nc), slice(3)), p[:, [0, 3, 4]])
    new_tet = bm.set_at(new_tet, (slice(2 * nc, 3 * nc), slice(3)), p[:, [1, 5, 3]])
    new_tet = bm.set_at(new_tet, (slice(3 * nc, 4 * nc), slice(3)), p[:, [2, 4, 5]])

    l = bm.zeros((nc, 3), dtype=block.positions.dtype, device=bm.get_device(block.positions))
    node = block.positions
    l = bm.set_at(l, (slice(None), 0), bm.sum((node[p[:, 0]] - node[p[:, 5]]) ** 2, axis=1))
    l = bm.set_at(l, (slice(None), 1), bm.sum((node[p[:, 1]] - node[p[:, 4]]) ** 2, axis=1))
    l = bm.set_at(l, (slice(None), 2), bm.sum((node[p[:, 2]] - node[p[:, 3]]) ** 2, axis=1))

    idx = bm.argmin(l, axis=1)
    table = bm.array(
        [(1, 3, 4, 2, 5, 0), (0, 2, 5, 3, 4, 1), (0, 4, 5, 1, 3, 2)],
        dtype=tet.dtype,
        device=bm.get_device(tet),
    )
    t = table[idx]
    rows = bm.arange(nc, dtype=tet.dtype, device=bm.get_device(tet))
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 0), p[rows, t[:, 0]])
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 1), p[rows, t[:, 1]])
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(4 * nc, 5 * nc), 3), p[rows, t[:, 5]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 0), p[rows, t[:, 1]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 1), p[rows, t[:, 2]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(5 * nc, 6 * nc), 3), p[rows, t[:, 5]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 0), p[rows, t[:, 2]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 1), p[rows, t[:, 3]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(6 * nc, 7 * nc), 3), p[rows, t[:, 5]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 0), p[rows, t[:, 3]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 1), p[rows, t[:, 0]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 2), p[rows, t[:, 4]])
    new_tet = bm.set_at(new_tet, (slice(7 * nc, 8 * nc), 3), p[rows, t[:, 5]])

    _set_sector(block, "tet", new_tet)
    _rebuild_topology(block)


def uniform_refine_tetrahedron(block: MeshBlock, n: int = 1, returnim: bool = False, **kwargs) -> list | None:
    """Uniformly refine tetrahedral entities in a MeshBlock in place."""
    im = [] if returnim else None
    for _ in range(n):
        _ensure_topology(block, ("tet", "segment"))
        if returnim:
            edge = _sector(block, "segment")
            node = block.positions
            edge2new = _arange_like(len(node), len(node) + len(edge), _sector(block, "tet"))
            im.append(_prolongation_from_edges(node, edge, edge2new))
        _refine_tetrahedron_once(block)
    if returnim:
        im.reverse()
        return im
    return None


def _refine_prism_once(block: MeshBlock) -> None:
    _ensure_topology(block, ("prism", "segment"), ("prism", "quad"))
    prism = _sector(block, "prism")
    edge = _sector(block, "segment")
    quad = _sector(block, "quad")
    c2e = block.relations[("prism", "segment")].tgt_indices
    c2q = block.relations[("prism", "quad")].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nq = len(quad)
    nc = len(prism)

    edge_center = _barycenter(block, edge)
    quad_center = _barycenter(block, quad)
    e = c2e + nn
    q = c2q + nn + ne

    new_prism = _empty_like((8 * nc, 6), prism)
    new_prism = bm.set_at(new_prism, (slice(0, None, 8), slice(None)), bm.stack([prism[:, 0], e[:, 0], e[:, 2], e[:, 3], q[:, 0], q[:, 2]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(1, None, 8), slice(None)), bm.stack([e[:, 0], e[:, 1], e[:, 2], q[:, 0], q[:, 1], q[:, 2]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(2, None, 8), slice(None)), bm.stack([prism[:, 1], e[:, 1], e[:, 0], e[:, 4], q[:, 1], q[:, 0]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(3, None, 8), slice(None)), bm.stack([prism[:, 2], e[:, 2], e[:, 1], e[:, 5], q[:, 2], q[:, 1]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(4, None, 8), slice(None)), bm.stack([e[:, 3], q[:, 0], q[:, 2], prism[:, 3], e[:, 6], e[:, 8]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(5, None, 8), slice(None)), bm.stack([q[:, 0], q[:, 1], q[:, 2], e[:, 6], e[:, 7], e[:, 8]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(6, None, 8), slice(None)), bm.stack([e[:, 4], q[:, 1], q[:, 0], prism[:, 4], e[:, 7], e[:, 6]], axis=1))
    new_prism = bm.set_at(new_prism, (slice(7, None, 8), slice(None)), bm.stack([e[:, 5], q[:, 2], q[:, 1], prism[:, 5], e[:, 8], e[:, 7]], axis=1))

    _append_positions(block, edge_center, quad_center)
    _set_sector(block, "prism", new_prism)
    _rebuild_topology(block)


def uniform_refine_prism(block: MeshBlock, n: int = 1, returnim: bool = False, **kwargs) -> list | None:
    """Uniformly refine prism entities in a MeshBlock in place."""
    im = [] if returnim else None
    for _ in range(n):
        _refine_prism_once(block)
    if returnim:
        return im
    return None


def _refine_hexahedron_once(block: MeshBlock) -> None:
    _ensure_topology(block, ("hex", "segment"), ("hex", "quad"))
    hex_ = _sector(block, "hex")
    edge = _sector(block, "segment")
    quad = _sector(block, "quad")
    c2e = block.relations[("hex", "segment")].tgt_indices
    c2f = block.relations[("hex", "quad")].tgt_indices
    node = block.positions
    nn = len(node)
    ne = len(edge)
    nf = len(quad)
    nc = len(hex_)

    edge_center = _barycenter(block, edge)
    face_center = _barycenter(block, quad)
    cell_center = _barycenter(block, hex_)
    c2n = hex_
    c2e = c2e + nn
    c2f = c2f + nn + ne
    c2c = _arange_like(nn + ne + nf, nn + ne + nf + nc, hex_)

    cell = _empty_like((8 * nc, 8), hex_)
    assignments = [
        (0, [c2n[:, 0], c2e[:, 0], c2e[:, 3], c2f[:, 0], c2e[:, 4], c2f[:, 4], c2f[:, 2], c2c]),
        (1, [c2n[:, 1], c2e[:, 1], c2e[:, 0], c2f[:, 0], c2e[:, 5], c2f[:, 3], c2f[:, 4], c2c]),
        (2, [c2n[:, 3], c2e[:, 2], c2e[:, 1], c2f[:, 0], c2e[:, 7], c2f[:, 5], c2f[:, 3], c2c]),
        (3, [c2n[:, 2], c2e[:, 3], c2e[:, 2], c2f[:, 0], c2e[:, 6], c2f[:, 2], c2f[:, 5], c2c]),
        (4, [c2n[:, 4], c2e[:, 11], c2e[:, 8], c2f[:, 1], c2e[:, 4], c2f[:, 2], c2f[:, 4], c2c]),
        (5, [c2n[:, 5], c2e[:, 8], c2e[:, 9], c2f[:, 1], c2e[:, 5], c2f[:, 4], c2f[:, 3], c2c]),
        (6, [c2n[:, 7], c2e[:, 9], c2e[:, 10], c2f[:, 1], c2e[:, 7], c2f[:, 3], c2f[:, 5], c2c]),
        (7, [c2n[:, 6], c2e[:, 10], c2e[:, 11], c2f[:, 1], c2e[:, 6], c2f[:, 5], c2f[:, 2], c2c]),
    ]
    for offset, cols in assignments:
        cell = bm.set_at(cell, (slice(offset, None, 8), slice(None)), bm.stack(cols, axis=1))

    _append_positions(block, edge_center, face_center, cell_center)
    _set_sector(block, "hex", cell)
    _rebuild_topology(block)


def uniform_refine_hexahedron(
    block: MeshBlock,
    n: int = 1,
    surface=None,
    interface=None,
    returnim: bool = False,
    **kwargs,
) -> list | None:
    """Uniformly refine hexahedral entities in a MeshBlock in place."""
    im = [] if returnim else None
    for _ in range(n):
        _ensure_topology(block, ("hex", "segment"), ("hex", "quad"))
        if returnim:
            hex_ = _sector(block, "hex")
            edge = _sector(block, "segment")
            quad = _sector(block, "quad")
            node = block.positions
            nn = len(node)
            ne = len(edge)
            nf = len(quad)
            nc = len(hex_)
            shape = (nn + ne + nf + nc, nn)
            values = bm.ones(nn + 2 * ne + 4 * nf + 8 * nc, **bm.context(node))
            values = bm.set_at(values, bm.arange(nn, nn + 2 * ne), 0.5)
            values = bm.set_at(values, bm.arange(nn + 2 * ne, nn + 2 * ne + 4 * nf), 0.25)
            values = bm.set_at(values, bm.arange(nn + 2 * ne + 4 * nf, nn + 2 * ne + 4 * nf + 8 * nc), 0.125)
            i0 = bm.arange(nn, dtype=hex_.dtype, device=bm.get_device(hex_))
            i1 = _arange_like(nn, nn + ne, hex_)
            i2 = _arange_like(nn + ne, nn + ne + nf, hex_)
            i3 = _arange_like(nn + ne + nf, nn + ne + nf + nc, hex_)
            i = bm.concat((i0, i1, i1, i2, i2, i2, i2, i3, i3, i3, i3, i3, i3, i3, i3), axis=0)
            j = bm.concat((i0, edge[:, 0], edge[:, 1], quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3], hex_[:, 0], hex_[:, 1], hex_[:, 2], hex_[:, 3], hex_[:, 4], hex_[:, 5], hex_[:, 6], hex_[:, 7]), axis=0)
            im.append(csr_matrix((values, (i, j)), shape))
        _refine_hexahedron_once(block)
    if returnim:
        im.reverse()
        return im
    return None


_DISPATCH: dict[str, RefineFunc] = {
    "segment": uniform_refine_segment,
    "tri": uniform_refine_triangle,
    "quad": uniform_refine_quadrilateral,
    "tet": uniform_refine_tetrahedron,
    "prism": uniform_refine_prism,
    "hex": uniform_refine_hexahedron,
}


def uniform_refine(block: MeshBlock, n: int = 1, schema_name: str | None = None, **kwargs) -> list | None:
    """Uniformly refine a MeshBlock in place.

    This function dispatches by ``schema_name`` or, when omitted, by the first
    root entity name in ``block.root_entity_names``. Shape validation is left to
    the caller-facing API layer.
    """
    if schema_name is None:
        if not block.root_entity_names:
            raise ValueError("cannot refine a MeshBlock without root entities")
        schema_name = block.root_entity_names[0]
    try:
        refine = _DISPATCH[schema_name]
    except KeyError as exc:
        raise NotImplementedError(f"uniform refinement is not implemented for {schema_name!r}") from exc
    return refine(block, n=n, **kwargs)
