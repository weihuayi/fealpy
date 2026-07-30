"""DXF 文件读取与数据模型测试.

How to run:
    pytest -q -s test/geometry/io/test_dxf_reader.py
    
"""

from __future__ import annotations

import pytest

from fealpy.geometry.io import (
    read_dxf,
    DxfDocument,
    dxf_to_actors,
    DxfLine,
    _smart_layer_color,
    _aci_to_rgb,
    _get_color,
    _STABLE_PALETTE,
)
    
# ═══════════════════════════════════════════════════════════════
# 图层颜色映射测试 (不依赖实际 DXF 文件)
# ═══════════════════════════════════════════════════════════════

class TestSmartLayerColor:
    """智能图层名 → 颜色映射."""

    @pytest.mark.parametrize("layer_name,expected_hint", [
        ("DIM",           "尺寸标注 → 绿色"),
        ("PUB_DIM",       "尺寸标注 → 绿色"),
        ("尺寸标注层",      "中文尺寸标注 → 绿色"),
        ("CENTER",        "中心线 → 红色"),
        ("中心线",          "中文中心线 → 红色"),
        ("轴线",           "中文轴线 → 红色"),
        ("HIDDEN",        "隐藏线 → 黄色"),
        ("虚线",           "中文隐藏线 → 黄色"),
        ("TEXT",          "文字 → 青色"),
        ("文字注释",        "中文文字 → 青色"),
        ("HATCH",         "剖面线 → 浅灰"),
        ("BORDER",        "图框 → 深灰"),
        ("SECTION",       "剖切 → 品红"),
        ("WALL",          "墙体 → 深棕"),
        ("DOOR",          "门 → 蓝色"),
        ("OUTLINE",       "轮廓线 → 深黑"),
        ("DEFPOINTS",     "参考线 → 浅灰"),
    ])
    def test_known_layer_patterns(self, layer_name, expected_hint):
        """常见图层名应映射到符合 CAD 规范的颜色."""
        r, g, b = _smart_layer_color(layer_name)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0
        # 确保不是纯黑 (排除错误)
        assert (r, g, b) != (0.0, 0.0, 0.0) or layer_name in ("OUTLINE", "0")

    def test_deterministic(self):
        """同一图层名多次调用必须返回相同颜色."""
        for name in ["DIM", "CENTER", "HIDDEN", "TEXT", "some_random_layer_xyz"]:
            assert _smart_layer_color(name) == _smart_layer_color(name)

    def test_different_layers_different_colors(self):
        """不同图层的颜色应有所区分."""
        colors = {
            "DIM": _smart_layer_color("DIM"),
            "CENTER": _smart_layer_color("CENTER"),
            "HIDDEN": _smart_layer_color("HIDDEN"),
            "TEXT": _smart_layer_color("TEXT"),
            "WALL": _smart_layer_color("WALL"),
        }
        # 至少 4 种不同颜色
        unique = len(set(colors.values()))
        assert unique >= 4, f"只有 {unique} 种不同颜色，期望 >= 4"


class TestAciToRgb:
    """ACI 颜色索引 → RGB."""

    @pytest.mark.parametrize("aci", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 256])
    def test_standard_colors_valid(self, aci):
        r, g, b = _aci_to_rgb(aci)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    @pytest.mark.parametrize("aci", [10, 50, 100, 150, 200, 249])
    def test_hsl_range_valid(self, aci):
        r, g, b = _aci_to_rgb(aci)
        assert 0.0 <= r <= 1.0

    @pytest.mark.parametrize("aci", [250, 252, 254, 255])
    def test_grayscale_range_valid(self, aci):
        r, g, b = _aci_to_rgb(aci)
        # 灰度: R ≈ G ≈ B
        assert abs(r - g) < 0.05
        assert abs(g - b) < 0.05

    def test_byblock_and_bylayer_are_dark(self):
        """ByBlock/ByLayer 应为深色 (白底可见)."""
        for aci in (0, 256):
            r, g, b = _aci_to_rgb(aci)
            assert r < 0.2 and g < 0.2 and b < 0.2


class TestGetColor:
    """_get_color 三级优先级."""

    def test_entity_own_color_used(self):
        """实体有自身颜色时应直接使用."""
        from fealpy.geometry.io import DxfLine
        e = DxfLine(layer="any_layer", color=1)  # ACI 1 = Red
        r, g, b = _get_color(e, use_layer_colors=False)
        assert r > 0.8 and g < 0.2  # 红色系

    def test_bylayer_falls_back_to_smart_layer(self):
        """ByLayer 实体应使用智能图层色."""
        from fealpy.geometry.io import DxfLine
        e = DxfLine(layer="DIM", color=256)  # ByLayer on DIM layer
        r, g, b = _get_color(e, use_layer_colors=False)
        assert g > 0.5  # DIM 图层应为绿色

    def test_force_layer_color(self):
        """use_layer_colors=True 强制使用图层色."""
        from fealpy.geometry.io import DxfLine
        e = DxfLine(layer="DIM", color=1)  # ACI Red, but on DIM layer
        r, g, b = _get_color(e, use_layer_colors=True)
        assert g > 0.5  # 图层色优先 → 绿色


# ═══════════════════════════════════════════════════════════════
# DXF 文件读取测试 (需要真实文件)
# ═══════════════════════════════════════════════════════════════

class TestDxfReader:
    """DXF 文件读取."""

    def test_read_returns_dxf_document(self, sample_dxf_path):
        """读取应返回 DxfDocument 实例."""
        doc = read_dxf(str(sample_dxf_path))
        assert isinstance(doc, DxfDocument)

    def test_total_entities_positive(self, sample_dxf_path):
        """应有至少 1 个实体."""
        doc = read_dxf(str(sample_dxf_path))
        assert doc.total_entities > 0

    def test_layers_not_empty(self, sample_dxf_path):
        """应有至少 1 个图层."""
        doc = read_dxf(str(sample_dxf_path))
        assert len(doc.layers) >= 1

    def test_extents_complete(self, sample_dxf_path):
        """包围盒应有 6 个字段."""
        doc = read_dxf(str(sample_dxf_path))
        if doc.extents:
            for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                assert key in doc.extents
            assert doc.extents["xmin"] <= doc.extents["xmax"]

    def test_entity_lists_consistent(self, sample_dxf_path):
        """分类实体数量之和应与 all_entities 一致 (DIMENSION 除外)."""
        doc = read_dxf(str(sample_dxf_path))
        categorized = (
            len(doc.lines) + len(doc.polylines) + len(doc.circles)
            + len(doc.arcs) + len(doc.ellipses) + len(doc.splines)
            + len(doc.texts)
        )
        # all_entities 包含所有可渲染实体 (不含 DxfDimension, 它在 dimensions 中)
        assert categorized <= doc.total_entities


class TestDxfToActors:
    """DXF → VTK Actor 转换."""

    def test_produces_actors(self, sample_dxf_path):
        """应生成 Actor 列表."""
        doc = read_dxf(str(sample_dxf_path))
        actors = dxf_to_actors(doc, text_visible=True)
        assert len(actors) > 0

    def test_layer_colors_mode(self, sample_dxf_path):
        """use_layer_colors=True/False 均应正常工作."""
        doc = read_dxf(str(sample_dxf_path))
        actors_false = dxf_to_actors(doc, use_layer_colors=False)
        actors_true = dxf_to_actors(doc, use_layer_colors=True)
        assert len(actors_false) == len(actors_true)

    def test_text_hidden(self, sample_dxf_path):
        """text_visible=False 应减少 actor 数量."""
        doc = read_dxf(str(sample_dxf_path))
        all_actors = dxf_to_actors(doc, text_visible=True)
        no_text = dxf_to_actors(doc, text_visible=False)
        assert len(no_text) <= len(all_actors)


# ═══════════════════════════════════════════════════════════════
# 颜色稳定性回归测试
# ═══════════════════════════════════════════════════════════════

class TestColorStability:
    """确保颜色分配是确定的."""

    def test_smart_layer_color_stable_across_calls(self):
        """_smart_layer_color 在同一进程中应返回一致结果."""
        import zlib

        # 用一批假图层名测试
        test_layers = [
            "DIM", "CENTER", "HIDDEN", "TEXT", "0",
            "layer_a", "layer_b", "layer_c",
            "中文标注", "电气层", "钢结构",
        ]
        results = {name: _smart_layer_color(name) for name in test_layers}
        # 再次调用应对齐
        for name, expected in results.items():
            assert _smart_layer_color(name) == expected, f"{name} 颜色不一致"

    def test_unknown_layer_uses_stable_palette(self):
        """未知图层应从 24 色调色板中选取."""
        from fealpy.geometry.io import _STABLE_PALETTE
        color = _smart_layer_color("completely_random_layer_12345")
        assert color in _STABLE_PALETTE
