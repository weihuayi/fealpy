import numpy as np

from fealpy.backend import bm
from fealpy.mesh.schema import QuadrilateralSchema
from fealpy.mesh.schema.entity_schema import EntityContext
from fealpy.mesh.storage import EntitySector, MeshBlock


def make_context(positions, quads):
    block = MeshBlock(positions=bm.asarray(positions, dtype=bm.float64))
    sector = EntitySector(schema_name="quad", indices=bm.asarray(quads, dtype=bm.int32))
    block.add_sector(sector, root=True)
    return EntityContext(block=block, sector=sector)


def test_quadrilateral_schema_barycenter_and_measure_2d():
    # Test 2D methods: barycenter/measure/normal/grad_lambda/tangent/bc_to_point.
    ctx = make_context(
        positions=[
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.5, 1.0]],
        quads=[
            [0, 1, 3, 2],
            [4, 5, 7, 6]])
    barycenter = np.asarray(QuadrilateralSchema.barycenter(ctx, None))
    measure = np.asarray(QuadrilateralSchema.measure(ctx, None))

    np.testing.assert_allclose(barycenter, np.array([[1.0, 0.5], [0.625, 0.5]]))
    np.testing.assert_allclose(measure, np.array([2.0, 1.25]))
    np.testing.assert_allclose(
        np.asarray(QuadrilateralSchema.normal(ctx, None)),
        np.zeros((2, 0, 2)),
    )
    np.testing.assert_allclose(np.asarray(QuadrilateralSchema.grad_lambda(ctx, np.array([0]))), np.array([[[-0.25, -0.5], [0.25, -0.5], [0.25, 0.5], [-0.25, 0.5]]]))
    np.testing.assert_allclose(np.asarray(QuadrilateralSchema.tangent(ctx, np.array([0]))), np.array([[[2.0, 0.0], [0.0, 1.0]]]))

    bc0 = bm.asarray([[0.5, 0.5]], dtype=bm.float64)
    bc1 = bm.asarray([[0.5, 0.5]], dtype=bm.float64)
    point = np.asarray(QuadrilateralSchema.bc_to_point(ctx, (bc0, bc1), np.array([0])))
    np.testing.assert_allclose(point, np.array([[[1.0, 0.5]]]))


def test_quadrilateral_schema_measure_3d_and_index():
    # Test 3D methods: measure/barycenter/normal/grad_lambda/tangent/bc_to_point.
    ctx = make_context(
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0]],
        quads=[
            [0, 1, 3, 2],
            [4, 5, 7, 6]])

    measure = np.asarray(QuadrilateralSchema.measure(ctx, None))
    barycenter = np.asarray(QuadrilateralSchema.barycenter(ctx, slice(1, 2)))
    normal = np.asarray(QuadrilateralSchema.normal(ctx, None))
    grad_lambda = np.asarray(QuadrilateralSchema.grad_lambda(ctx, slice(1, 2)))
    bc0 = bm.asarray([[0.5, 0.5]], dtype=bm.float64)
    bc1 = bm.asarray([[0.5, 0.5]], dtype=bm.float64)
    point = np.asarray(QuadrilateralSchema.bc_to_point(ctx, (bc0, bc1), slice(1, 2)))

    np.testing.assert_allclose(measure, np.array([np.sqrt(2.0), 2.0]))
    np.testing.assert_allclose(barycenter, np.array([[1.0, 0.5, 0.0]]))
    np.testing.assert_allclose(
        normal,
        np.array([[[0.0, -2.0, 2.0]], [[0.0, 0.0, 4.0]]]),
    )
    np.testing.assert_allclose(grad_lambda, np.array([[[-0.25, -0.5, 0.0], [0.25, -0.5, 0.0], [0.25, 0.5, 0.0], [-0.25, 0.5, 0.0]]]))
    np.testing.assert_allclose(np.asarray(QuadrilateralSchema.tangent(ctx, slice(1, 2))), np.array([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
    np.testing.assert_allclose(point, np.array([[[1.0, 0.5, 0.0]]]))
    assert QuadrilateralSchema.geo_dimension(ctx) == 3


def test_quadrilateral_schema_multi_index():
    # Test index helpers: multi_index/multi_index_sort/num_multi_index.
    mi = np.asarray(QuadrilateralSchema.multi_index((2,)))
    rect_mi = np.asarray(QuadrilateralSchema.multi_index((1, 2)))

    np.testing.assert_array_equal(mi, np.array([
        [4, 0, 0, 0],
        [2, 2, 0, 0],
        [0, 4, 0, 0],
        [2, 0, 2, 0],
        [1, 1, 1, 1],
        [0, 2, 0, 2],
        [0, 0, 4, 0],
        [0, 0, 2, 2],
        [0, 0, 0, 4],
    ], dtype=np.int32))
    np.testing.assert_array_equal(rect_mi, np.array([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
    ], dtype=np.int32))
    assert QuadrilateralSchema.num_multi_index((2,)) == 9
    assert QuadrilateralSchema.num_multi_index((1, 2)) == 6


def test_quadrilateral_schema_quadrature_formula_shape():
    qf = QuadrilateralSchema.quadrature_formula(2)
    bcs, ws = qf.get_quadrature_points_and_weights()

    assert isinstance(bcs, tuple)
    assert len(bcs) == 2
    assert bcs[0].shape == (2, 2)
    assert bcs[1].shape == (2, 2)
    assert ws.shape == (4, )


def test_quadrilateral_schema_quadrature_formula_2d():
    # Test 2D quadrature path: quadrature_formula/bc_to_point.
    ctx = make_context(positions=[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], quads=[[0, 1, 2, 3]])
    qf = QuadrilateralSchema.quadrature_formula(2)
    bcs, ws = qf.get_quadrature_points_and_weights()
    point = np.asarray(QuadrilateralSchema.bc_to_point(ctx, bcs, np.array([0])))
    assert isinstance(bcs, tuple)
    assert len(bcs) == 2
    assert bcs[0].shape == (2, 2)
    assert bcs[1].shape == (2, 2)
    assert ws.shape == (4, )
    np.testing.assert_allclose(point.shape, np.array([1, 4, 2]))
