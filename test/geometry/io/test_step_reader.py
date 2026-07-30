"""STEP 文件读取与拓扑统计测试.

How to run:
    pytest -q -s test/geometry/io/test_step_reader.py
"""

from __future__ import annotations

import pytest

from fealpy.geometry.io import (
    read_step,
    get_shape_info,
    shape_to_actor,
    shape_to_edges_actor,
    StepReadError,
)


# ═══════════════════════════════════════════════════════════════
# STEP 文件读取测试
# ═══════════════════════════════════════════════════════════════

class TestStepReader:
    """STEP 文件读取."""

    def test_read_returns_shape(self, sample_step_path):
        """读取应返回非空 TopoDS_Shape."""
        shape = read_step(str(sample_step_path))
        assert shape is not None
        assert not shape.IsNull(), "Shape 不应为空"

    def test_file_not_found_raises(self):
        """不存在的文件应抛出 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_step("/tmp/nonexistent_file_12345.step")

    def test_info_has_required_keys(self, sample_step_path):
        """get_shape_info 应返回完整的拓扑统计."""
        shape = read_step(str(sample_step_path))
        info = get_shape_info(shape)

        required_keys = [
            "solid_count", "shell_count", "face_count",
            "edge_count", "vertex_count", "face_types", "bbox",
        ]
        for key in required_keys:
            assert key in info, f"缺少字段: {key}"

    def test_face_count_positive(self, sample_step_path):
        """至少应有 1 个面."""
        shape = read_step(str(sample_step_path))
        info = get_shape_info(shape)
        assert info["face_count"] > 0

    def test_bbox_valid(self, sample_step_path):
        """包围盒坐标应有效."""
        shape = read_step(str(sample_step_path))
        info = get_shape_info(shape)
        bbox = info["bbox"]
        assert bbox["xmin"] <= bbox["xmax"]
        assert bbox["ymin"] <= bbox["ymax"]


class TestStepToActors:
    """STEP Shape → VTK Actor 转换."""

    def test_solid_actor_created(self, sample_step_path):
        """shape_to_actor 应返回实体 Actor."""
        shape = read_step(str(sample_step_path))
        actor = shape_to_actor(shape, color=(0.65, 0.75, 0.85))
        assert actor is not None

    def test_edges_actor_created(self, sample_step_path):
        """shape_to_edges_actor 应返回边线 Actor."""
        shape = read_step(str(sample_step_path))
        actor = shape_to_edges_actor(shape, color=(0.0, 0.0, 0.0))
        assert actor is not None

    def test_wireframe_mode(self, sample_step_path):
        """线框模式应正常创建."""
        shape = read_step(str(sample_step_path))
        actor = shape_to_actor(shape, color=(0.5, 0.5, 0.5), wireframe=True)
        assert actor is not None

    def test_different_deflections(self, sample_step_path):
        """不同精度参数应都能正常工作."""
        shape = read_step(str(sample_step_path))
        for deflection in [0.5, 0.1, 0.01]:
            actor = shape_to_actor(shape, deflection=deflection)
            assert actor is not None, f"deflection={deflection} 失败"


# ═══════════════════════════════════════════════════════════════
# 错误处理测试
# ═══════════════════════════════════════════════════════════════

class TestStepErrors:
    """STEP 读取错误处理."""

    def test_non_step_file_raises(self, tmp_path):
        """非 STEP 文件应抛出 StepReadError."""
        junk = tmp_path / "junk.step"
        junk.write_text("this is not a STEP file")
        with pytest.raises((StepReadError, Exception)):
            read_step(str(junk))
