# 文件位置: tests/mesh/unit/schema/test_point_schema.py

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import PointSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh


def _as_list(value):
    return bm.to_numpy(value).tolist()


def _assert_allclose(actual, expected, message):
    assert bm.allclose(actual, expected), (
        f"{message}\n"
        f"actual: {_as_list(actual)}\n"
        f"expected: {_as_list(expected)}"
    )


def _assert_equal(actual, expected, message):
    actual_list = _as_list(actual)
    expected_list = _as_list(expected)
    assert actual_list == expected_list, (
        f"{message}\n"
        f"actual: {actual_list}\n"
        f"expected: {expected_list}"
    )


def _assert_shape(actual, expected_shape, message):
    assert actual.shape == expected_shape, (
        f"{message}\n"
        f"actual shape: {actual.shape}\n"
        f"expected shape: {expected_shape}"
    )


def _build_point_view():
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=bm.float64,
    )
    point_sector = EntitySector(
        schema_name="point",
        indices=bm.asarray([0, 2], dtype=bm.int64),
    )
    block = MeshBlock(positions=positions)
    block.add_sector(point_sector, root=True)
    mesh = Mesh(block)
    return mesh, mesh.Entity("point")


class TestPointSchema:
    """
    PointSchema 单元测试。

    测试通过用户入口 Mesh.sector("point") 获得 EntityView，再验证
    PointSchema 的核心算法是否符合 mesh_05_algorithm_migration 的接口合同。
    """

    def test_schema_dispatch_and_attributes(self):
        """
        [结构验证]：用户从 Mesh.sector("point") 获取的实体视图必须分派到 PointSchema。
        同时验证 node 的拓扑维数、几何维数和 ccw 元数据。
        """
        mesh, point_view = _build_point_view()

        assert point_view.schema is PointSchema, (
            "User entry Mesh.sector('point') should dispatch node algorithms to PointSchema"
        )
        assert point_view.size() == 2, "Node sector contains exactly the two selected node entities"
        assert point_view.top_dimension() == 0, "PointSchema is a 0D entity schema"
        assert PointSchema.OFace == {}, "Point has no sub-entities, so OFace must be an empty dict"
        assert PointSchema.SFace == {}, "Point has no sub-entities, so SFace must be an empty dict"
        assert point_view.geo_dimension() == mesh.geo_dimension() == 3, (
            "Node geometric dimension must equal positions.shape[1]"
        )

    def test_barycenter_through_user_view(self):
        """
        [几何算法验证]：点实体的重心就是点坐标本身。
        用户入口 point_view.barycenter() 应返回 positions[point_view.indices]。
        """
        _, point_view = _build_point_view()

        expected = bm.asarray(
            [
                [0.0, 0.0, 0.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=bm.float64,
        )
        _assert_allclose(
            point_view.barycenter(),
            expected,
            "A 0D node entity is its own barycenter; expected positions[point_view.indices]",
        )

    def test_measure_through_user_view(self):
        """
        [几何算法验证]：新 schema 语义下，0 维实体测度为 1。
        """
        _, point_view = _build_point_view()

        _assert_allclose(
            point_view.measure(),
            bm.ones((point_view.size(),), dtype=bm.float64),
            "0D entity measure is 1 for each node under new schema semantics",
        )

    def test_quadrature_formula_via_schema_behind_user_view(self):
        """
        [积分公式验证]：点实体的 0 维求积公式只有一个重心坐标点 [1]，权重为 1。
        当前 EntityView 尚未包装 quadrature_formula，因此通过 point_view.schema 验证。
        """
        _, point_view = _build_point_view()

        qf = point_view.schema.quadrature_formula(1, qtype=None)
        bcs, weights = qf.get_quadrature_points_and_weights()

        assert isinstance(bcs, tuple), "Handoff requires bcs to be a tuple of tensors"
        assert len(qf) == 1, "Point quadrature has exactly one quadrature point"
        _assert_equal(
            bcs[0],
            bm.asarray([[1.0]], dtype=bm.float64),
            "Point quadrature barycentric coordinate must be [1]",
        )
        _assert_equal(
            weights,
            bm.asarray([1.0], dtype=bm.float64),
            "Point quadrature weight must be 1",
        )

    def test_grad_lambda_through_user_view(self):
        """
        [几何算法验证]：点上唯一重心坐标恒为 1，因此梯度为 0。
        返回形状应为 (N, 1, GD)。
        """
        _, point_view = _build_point_view()
        gd = point_view.geo_dimension()

        grad = point_view.grad_lambda()
        _assert_shape(
            grad,
            (point_view.size(), 1, gd),
            "Node has one barycentric coordinate and GD cartesian directions",
        )
        _assert_allclose(
            grad,
            bm.zeros((point_view.size(), 1, gd), dtype=bm.float64),
            "The only node barycentric coordinate is constant 1, so its gradient is 0",
        )

    def test_normal_and_tangent_through_user_view(self):
        """
        [维度语义验证]：按照 handoff 约定，法向数量为 G - T，切向数量为 T。
        对 node 而言 T=0，因此 normal 形状为 (N, GD, GD)，tangent 形状为 (N, 0, GD)。
        """
        _, point_view = _build_point_view()
        nnode = point_view.size()
        gd = point_view.geo_dimension()
        top_dim = point_view.top_dimension()

        normal = point_view.normal()
        _assert_shape(
            normal,
            (nnode, gd - top_dim, gd),
            "Handoff rule: normal return shape is [entity_count, G - T, G]",
        )
        _assert_allclose(
            normal,
            bm.broadcast_to(bm.eye(gd, dtype=bm.float64), (nnode, gd, gd)),
            "For a node T=0, the normal space is the full ambient space standard basis",
        )

        tangent = point_view.tangent()
        _assert_shape(
            tangent,
            (nnode, top_dim, gd),
            "Handoff rule: tangent return shape is [entity_count, T, G]; node has T=0",
        )

    def test_bc_to_point_via_schema_behind_user_view(self):
        """
        [几何算法验证]：点的合法重心坐标只能是 [1]。
        当前 EntityView 尚未包装 bc_to_point，因此通过 point_view.schema 验证背后的 schema 方法。
        """
        _, point_view = _build_point_view()
        ctx = point_view.context()

        bcs = (bm.asarray([[1.0], [1.0]], dtype=bm.float64),)
        points = point_view.schema.bc_to_point(ctx, bcs, None)
        expected = bm.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
            ],
            dtype=bm.float64,
        )

        _assert_shape(
            points,
            (point_view.size(), 2, point_view.geo_dimension()),
            "bc_to_point maps two barycentric samples for each node entity",
        )
        _assert_allclose(
            points,
            expected,
            "Node barycentric coordinate [1] must map back to the node coordinates",
        )

    def test_bc_to_point_rejects_invalid_node_barycentric_value(self):
        """
        [接口合同验证]：点实体的重心坐标必须恒为 [1]，非法值不能被映射为物理点。
        """
        _, point_view = _build_point_view()
        ctx = point_view.context()
        invalid_bcs = (bm.asarray([[0.5]], dtype=bm.float64),)

        with pytest.raises(ValueError):
            point_view.schema.bc_to_point(ctx, invalid_bcs, None)


    def test_multi_index_via_schema_behind_user_view(self):
        """
        [多重指标验证]：node 是一个顶点的退化单纯形，次数 p 只有一个指标 [p]。
        """
        _, point_view = _build_point_view()

        _assert_equal(
            point_view.schema.multi_index((3,)),
            bm.asarray([[3]], dtype=bm.int32),
            "Node is the one-vertex simplex degeneration; degree p has exactly one index [p]",
        )

    def test_multi_index_rejects_scalar_p(self):
        """
        [接口合同验证]：handoff 要求 multi_index 的 p 参数必须是整数元组。
        """
        _, point_view = _build_point_view()

        with pytest.raises(TypeError):
            point_view.schema.multi_index(3)
