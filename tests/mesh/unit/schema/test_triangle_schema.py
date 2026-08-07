import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import TriangleSchema
from fealpy.mesh.schema.entity_schema import EntityContext
from fealpy.mesh.storage import MeshBlock, EntitySector
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.ipoints import MultiIndex as _MI
from fealpy.quadrature.triangle import TriangleQuadrature


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return bm.to_numpy(value)


def make_context(points):
    pts = bm.tensor(points, dtype=bm.float64)
    tri = bm.tensor([[0, 1, 2]], dtype=bm.int32)
    block = MeshBlock(positions=pts)
    sector = EntitySector(schema_name="tri", indices=tri)
    block.add_sector(sector, root=True)
    return EntityContext(block, sector)


def make_topology_context(points, cells):
    pts = bm.tensor(points, dtype=bm.float64)
    tri = bm.tensor(cells, dtype=bm.int32)
    block = MeshBlock(positions=pts)
    sector = EntitySector(schema_name="tri", indices=tri)
    block.add_sector(sector, root=True)
    TopologyBuilder.construct(block)
    return EntityContext(block, sector)


def test_triangle_schema_geo_barycenter_measure_2d():
    ctx = make_context(np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64))

    bary = to_numpy(TriangleSchema.barycenter(ctx, None))
    np.testing.assert_allclose(bary, np.array([[1.0 / 3.0, 1.0 / 3.0]]))

    measure = to_numpy(TriangleSchema.measure(ctx, None))
    np.testing.assert_allclose(measure, np.array([0.5]))

    assert TriangleSchema.geo_dimension(ctx) == 2


def test_triangle_schema_grad_lambda_2d():
    ctx = make_context(np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64))

    grads = to_numpy(TriangleSchema.grad_lambda(ctx, None))
    expected = np.array([
        [-1.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    np.testing.assert_allclose(grads[0], expected)


def test_triangle_schema_bc_to_point():
    ctx = make_context(np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0]
    ], dtype=np.float64))
    bc = bm.tensor([[0.25, 0.25, 0.5]], dtype=bm.float64)
    point = to_numpy(TriangleSchema.bc_to_point(ctx, (bc,), None))
    expected = np.array([[[0.5, 1.0]]])
    np.testing.assert_allclose(point, expected)


def test_triangle_schema_tangent_and_normal_2d():
    ctx = make_context(np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float64))

    tangent = to_numpy(TriangleSchema.tangent(ctx, None))
    expected_tangent = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    np.testing.assert_allclose(tangent, expected_tangent)

    normal = to_numpy(TriangleSchema.normal(ctx, None))
    assert normal.shape == (1, 0, 2)
    assert normal.size == 0


def test_triangle_schema_normal_3d():
    ctx = make_context(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64))

    normal = to_numpy(TriangleSchema.normal(ctx, None))
    assert normal.shape == (1, 1, 3)
    np.testing.assert_allclose(normal[0, 0], np.array([0.0, 0.0, 1.0]))


def test_triangle_schema_multi_index():
    order = (2,)
    mi = to_numpy(TriangleSchema.multi_index(order))
    expected = to_numpy(_MI.multi_index_matrix(order[0], 3))
    np.testing.assert_array_equal(mi, expected)

    inner = to_numpy(TriangleSchema.multi_index(order, internal=True))
    expected_inner = to_numpy(_MI.multi_index_inner(order[0], 3))
    np.testing.assert_array_equal(inner, expected_inner)

    assert TriangleSchema.num_multi_index(order) == expected.shape[0]
    assert TriangleSchema.num_multi_index(order, internal=True) == expected_inner.shape[0]


def test_triangle_schema_quadrature():
    qf = TriangleSchema.quadrature_formula(3)
    assert isinstance(qf, TriangleQuadrature)
    assert qf.number_of_quadrature_points() > 0


def test_triangle_schema_local_entity_relation_size_and_boundary():
    ctx = make_topology_context(
        np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float64),
        np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.int32),
    )

    assert TriangleSchema.size(ctx) == 2
    assert TriangleSchema.local_entity("segment") == [[1, 2], [2, 0], [0, 1]]
    assert TriangleSchema.local_entity("point") == [[0], [1], [2]]

    edge_relation = TriangleSchema.relation(ctx, "segment")
    edge_sector = to_numpy(ctx.block.get_sector("segment").indices)
    related_edges = edge_sector[to_numpy(edge_relation.tgt_indices)]
    expected_edges = to_numpy(ctx.sector.indices)[:, TriangleSchema.local_entity("segment")]
    np.testing.assert_array_equal(
        np.sort(related_edges, axis=-1),
        np.sort(expected_edges, axis=-1),
    )

    node_relation = TriangleSchema.relation(ctx, "point")
    point_sector = to_numpy(ctx.block.get_sector("point").indices).reshape(-1)
    related_nodes = point_sector[to_numpy(node_relation.tgt_indices)]
    np.testing.assert_array_equal(related_nodes, to_numpy(ctx.sector.indices))

    boundary = TriangleSchema.boundary(ctx)
    np.testing.assert_array_equal(to_numpy(boundary.index), np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(to_numpy(boundary.mask), np.array([True, True]))


def test_triangle_schema_scalar_index_matches_slice_in_3d():
    ctx = make_topology_context(
        np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64),
        np.array([
            [0, 1, 2],
            [0, 1, 3],
        ], dtype=np.int32),
    )

    np.testing.assert_allclose(to_numpy(TriangleSchema.measure(ctx, None)), np.array([3.0, 1.0]))

    for method, args in [
        (TriangleSchema.barycenter, ()),
        (TriangleSchema.grad_lambda, ()),
        (TriangleSchema.measure, ()),
        (TriangleSchema.normal, ()),
        (TriangleSchema.tangent, ()),
        (TriangleSchema.bc_to_point, ((bm.tensor([[0.25, 0.25, 0.5]], dtype=bm.float64),),)),
    ]:
        full = to_numpy(method(ctx, *args, slice(None)))
        selected = to_numpy(method(ctx, *args, 1))
        np.testing.assert_allclose(selected, full[1:2])
