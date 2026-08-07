"""
DXF 文件读取、数据模型与 VTK 数据转换.

读取 DXF 文件 → DxfDocument 结构化数据,
并将各类实体转换为 VTK Actor / vtkPolyData 渲染对象.
"""

from __future__ import annotations

import os
import math
import zlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import vtk

logger = logging.getLogger(__name__)


# ============================================================
# DXF 数据模型 (dataclasses)
# ============================================================

@dataclass
class DxfEntity:
    """DXF 实体统一基类 — 所有几何/文字实体都继承自此类."""
    layer: str = "0"
    color: int = 7
    entity_type: str = ""  # dxftype: LINE, CIRCLE, TEXT, ...


@dataclass
class DxfLine(DxfEntity):
    """DXF 线段"""
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DxfPolyline(DxfEntity):
    """DXF 多段线"""
    points: List[Tuple[float, float]] = field(default_factory=list)
    closed: bool = False


@dataclass
class DxfCircle(DxfEntity):
    """DXF 圆"""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0


@dataclass
class DxfArc(DxfEntity):
    """DXF 圆弧"""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    start_angle: float = 0.0   # 度
    end_angle: float = 360.0   # 度


@dataclass
class DxfEllipse(DxfEntity):
    """DXF 椭圆"""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    major_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    ratio: float = 1.0
    start_angle: float = 0.0
    end_angle: float = 360.0


@dataclass
class DxfSpline(DxfEntity):
    """DXF 样条曲线 (BSpline.approximate 采样后的点列)"""
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    degree: int = 3


@dataclass
class DxfText(DxfEntity):
    """DXF 文字 (TEXT / MTEXT / ATTRIB / ATTDEF 统一)"""
    text: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    height: float = 2.5
    rotation: float = 0.0  # 旋转角 (度)
    width_factor: float = 1.0
    horizontal_alignment: int = 0
    vertical_alignment: int = 0
    attachment_point: int = 0  # MTEXT: 1..9
    style: str = "Standard"


@dataclass
class DxfDimension(DxfEntity):
    """DXF 尺寸标注 (已通过 virtual_entities 展开, 保留文本用于显示)"""
    text: str = ""
    text_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    text_height: float = 2.5


@dataclass
class DxfDocument:
    """DXF 文档数据模型 — 包含所有可提取的图形实体"""
    # 按类型分组的实体列表
    lines: List[DxfLine] = field(default_factory=list)
    polylines: List[DxfPolyline] = field(default_factory=list)
    circles: List[DxfCircle] = field(default_factory=list)
    arcs: List[DxfArc] = field(default_factory=list)
    ellipses: List[DxfEllipse] = field(default_factory=list)
    splines: List[DxfSpline] = field(default_factory=list)
    texts: List[DxfText] = field(default_factory=list)
    dimensions: List[DxfDimension] = field(default_factory=list)

    # 全部实体的统一列表 (DxfEntity 基类, 方便统一遍历)
    all_entities: List[DxfEntity] = field(default_factory=list)

    # 文档元数据
    layers: List[str] = field(default_factory=list)
    block_names: List[str] = field(default_factory=list)
    extents: Optional[Dict[str, float]] = None
    source_dimension_count: int = 0
    insert_attribute_count: int = 0

    @property
    def total_entities(self) -> int:
        return len(self.all_entities)


# ============================================================
# DXF 读取异常
# ============================================================

class DxfReadError(Exception):
    """DXF 文件读取异常"""
    pass


# ============================================================
# DXF 读取内部辅助函数
# ============================================================

def _clean_dim_fmt(raw: str) -> str:
    """清理 DXF 尺寸标注文字中的格式代码 (保留可读文本)."""
    import re
    if not raw:
        return ""
    s = re.sub(r'\\A[012];', '', raw)
    s = re.sub(r'\{[^}]*?;', '', s)
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\\S\s*([^^;]*)\^([^;]*);', r'  \1/\2', s)
    for code, sym in [('%%c', '⌀'), ('%%C', '⌀'), ('%%d', '°'), ('%%D', '°'), ('%%p', '±'), ('%%P', '±')]:
        s = s.replace(code, sym)
    return re.sub(r'\s+', ' ', s).strip()


def _to_tuple_3d(pt) -> Tuple[float, float, float]:
    """将 ezdxf Vec3 / numpy array / tuple 统一转为 (x, y, z)."""
    if hasattr(pt, 'x'):
        return (pt.x, pt.y, pt.z)
    return (float(pt[0]), float(pt[1]), float(pt[2]))


def _sample_arc_bbox(
    cx: float, cy: float, cz: float, r: float,
    start_deg: float, end_deg: float,
) -> List[Tuple[float, float, float]]:
    """对圆弧采样关键点以计算精确包围盒."""
    import math
    pts = [
        (cx + r * math.cos(math.radians(start_deg)),
         cy + r * math.sin(math.radians(start_deg)), cz),
        (cx + r * math.cos(math.radians(end_deg)),
         cy + r * math.sin(math.radians(end_deg)), cz),
    ]
    s, e = start_deg, end_deg
    if e <= s:
        e += 360.0
    for q in [0.0, 90.0, 180.0, 270.0, 360.0]:
        if s <= q <= e:
            pts.append((cx + r * math.cos(math.radians(q)),
                        cy + r * math.sin(math.radians(q)), cz))
    return pts


# ============================================================
# DXF 文件读取
# ============================================================

def read_dxf(filepath: str, debug: bool = False) -> DxfDocument:
    """
    读取 DXF 文件，返回结构化的 DxfDocument.

    架构: 遍历 ModelSpace 中每一个实体 (for entity in msp),
    按 dxftype 分派到对应 Parser.
    INSERT / DIMENSION / LEADER → virtual_entities() 展开为基本图元.
    SPLINE → BSpline.approximate() 采样为多段线点列.
    TEXT / MTEXT / ATTRIB / ATTDEF → 统一为 DxfText.

    Parameters
    ----------
    filepath : str
        DXF 文件路径.
    debug : bool
        开启后打印每个实体的解析日志 (handle, type, content, position).
    """
    import ezdxf

    file_path = Path(filepath).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"DXF 文件不存在: {filepath}")

    try:
        dxf_doc = ezdxf.readfile(str(file_path))
    except Exception as e:
        raise DxfReadError(f"DXF 读取失败: {e}")

    msp = dxf_doc.modelspace()
    result = DxfDocument()
    result.layers = [layer.dxf.name for layer in dxf_doc.layers]
    result.block_names = [b.name for b in dxf_doc.blocks if not b.name.startswith("*")]

    all_pts: List[Tuple[float, float, float]] = []

    # debug 统计
    debug_handles_seen: Dict[str, int] = {}
    debug_type_count: Dict[str, int] = {}
    debug_text_log: List[str] = []

    def _add(x, y, z=0.0):
        all_pts.append((float(x), float(y), float(z)))

    def _get_handle(raw) -> str:
        try:
            return raw.dxf.handle
        except Exception:
            return "?"

    def _debug_log(raw, parsed_type: str, extra: str = ""):
        if not debug:
            return
        t = raw.dxftype()
        h = _get_handle(raw)
        debug_type_count[t] = debug_type_count.get(t, 0) + 1
        if h != "?":
            debug_handles_seen[h] = debug_handles_seen.get(h, 0) + 1
        if parsed_type in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            debug_text_log.append(f"[{parsed_type}] handle={h} {extra}")

    def _dispatch(e: DxfEntity):
        result.all_entities.append(e)
        t = e.entity_type
        if t in ("LINE",):
            result.lines.append(e)
        elif t in ("LWPOLYLINE", "POLYLINE", "SOLID", "TRACE"):
            result.polylines.append(e)
        elif t in ("CIRCLE", "POINT"):
            result.circles.append(e)
        elif t == "ARC":
            result.arcs.append(e)
        elif t == "ELLIPSE":
            result.ellipses.append(e)
        elif t == "SPLINE":
            result.splines.append(e)
        elif t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            result.texts.append(e)

    def _dimension_text_height(raw) -> float:
        try:
            style = dxf_doc.dimstyles.get(raw.dxf.dimstyle)
            height = float(style.dxf.dimtxt)
            scale = float(style.dxf.dimscale)
            if scale <= 0.0:
                scale = 1.0
            if height > 0.0:
                return height * scale
        except Exception:
            pass
        return 2.5

    def _dimension_fallback_text(raw) -> str:
        try:
            measurement = float(raw.get_measurement())
        except Exception:
            return ""

        dim_type = int(getattr(raw.dxf, "dimtype", 0)) & 15
        precision = 3
        try:
            style = dxf_doc.dimstyles.get(raw.dxf.dimstyle)
            precision = int(style.dxf.dimadec if dim_type in (2, 5) else style.dxf.dimdec)
            if dim_type not in (2, 5):
                measurement *= float(style.dxf.dimlfac)
        except Exception:
            pass

        value = f"{measurement:.{max(0, precision)}f}".rstrip("0").rstrip(".")
        raw_text = str(getattr(raw.dxf, "text", "") or "")
        if raw_text:
            return _clean_dim_fmt(raw_text).replace("<>", value)
        if dim_type == 3:
            return f"⌀{value}"
        if dim_type == 4:
            return f"R{value}"
        if dim_type in (2, 5):
            return f"{value}°"
        return value

    def _parse_entity(
        raw, depth: int = 0, default_text_height: Optional[float] = None,
    ) -> Optional[DxfEntity]:
        """解析单个 ezdxf 实体 → DxfEntity."""
        t = raw.dxftype()
        layer = raw.dxf.layer if hasattr(raw.dxf, 'layer') else "0"
        color = raw.dxf.color if hasattr(raw.dxf, 'color') else 7
        prefix = "  " * depth

        if t == "LINE":
            s, e = raw.dxf.start, raw.dxf.end
            _add(s.x, s.y, s.z)
            _add(e.x, e.y, e.z)
            _debug_log(raw, "LINE", f"({s.x:.1f},{s.y:.1f})→({e.x:.1f},{e.y:.1f})")
            return DxfLine(entity_type=t, layer=layer, color=color,
                           start=(s.x, s.y, s.z), end=(e.x, e.y, e.z))

        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = []
            try:
                with raw.points() as points:
                    for pt in points:
                        pts.append((pt[0], pt[1]))
                        _add(pt[0], pt[1])
            except Exception:
                pass
            if not pts:
                return None
            _debug_log(raw, "POLYLINE", f"{len(pts)}pts")
            return DxfPolyline(entity_type=t, layer=layer, color=color,
                               points=pts, closed=getattr(raw, 'closed', False))

        elif t == "CIRCLE":
            c = raw.dxf.center
            r = raw.dxf.radius
            _add(c.x - r, c.y - r, c.z)
            _add(c.x + r, c.y + r, c.z)
            _debug_log(raw, "CIRCLE", f"r={r:.2f} ({c.x:.1f},{c.y:.1f})")
            return DxfCircle(entity_type=t, layer=layer, color=color,
                             center=(c.x, c.y, c.z), radius=r)

        elif t == "ARC":
            c = raw.dxf.center
            r = raw.dxf.radius
            sa, ea = raw.dxf.start_angle, raw.dxf.end_angle
            for pt in _sample_arc_bbox(c.x, c.y, c.z, r, sa, ea):
                _add(*pt)
            _debug_log(raw, "ARC", f"r={r:.2f} {sa}°→{ea}°")
            return DxfArc(entity_type=t, layer=layer, color=color,
                          center=(c.x, c.y, c.z), radius=r,
                          start_angle=sa, end_angle=ea)

        elif t == "ELLIPSE":
            c = raw.dxf.center
            maj = raw.dxf.major_axis
            rx = math.sqrt(maj.x**2 + maj.y**2 + maj.z**2)
            ry = rx * raw.dxf.ratio
            _add(c.x - rx, c.y - ry, c.z)
            _add(c.x + rx, c.y + ry, c.z)
            _debug_log(raw, "ELLIPSE", f"ratio={raw.dxf.ratio:.2f}")
            return DxfEllipse(entity_type=t, layer=layer, color=color,
                              center=(c.x, c.y, c.z),
                              major_axis=(maj.x, maj.y, maj.z),
                              ratio=raw.dxf.ratio,
                              start_angle=getattr(raw.dxf, 'start_param', 0.0),
                              end_angle=getattr(raw.dxf, 'end_param', 360.0))

        elif t == "SPLINE":
            try:
                ct = raw.construction_tool()
                sampled = [tuple(map(float, p)) for p in ct.approximate(segments=200)]
                for p in sampled:
                    _add(*p)
                if len(sampled) >= 2:
                    _debug_log(raw, "SPLINE", f"{len(sampled)}pts deg={raw.dxf.degree}")
                    return DxfSpline(entity_type=t, layer=layer, color=color,
                                     points=sampled, degree=raw.dxf.degree)
            except Exception:
                pass
            return None

        elif t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            try:
                text = raw.plain_text() if hasattr(raw, 'plain_text') else str(raw.dxf.text)
            except Exception:
                text = ""
            if not text or not text.strip():
                return None
            ins = raw.dxf.insert
            if t == "MTEXT":
                height = getattr(raw.dxf, "char_height", default_text_height or 2.5)
                attachment_point = int(getattr(raw.dxf, "attachment_point", 1))
                width_factor = 1.0
                try:
                    rotation = float(raw.get_rotation())
                except Exception:
                    rotation = float(getattr(raw.dxf, "rotation", 0.0))
            else:
                height = getattr(raw.dxf, "height", default_text_height or 2.5)
                attachment_point = 0
                width_factor = float(getattr(raw.dxf, "width", 1.0) or 1.0)
                rotation = float(getattr(raw.dxf, "rotation", 0.0))
            if not height or float(height) <= 0.0:
                height = default_text_height or 2.5

            halign = int(getattr(raw.dxf, "halign", 0))
            valign = int(getattr(raw.dxf, "valign", 0))
            position = (ins.x, ins.y, ins.z)
            align_point = getattr(raw.dxf, "align_point", None)
            if align_point is not None and (halign != 0 or valign != 0):
                position = _to_tuple_3d(align_point)

            _add(*position)
            content_preview = text[:40].replace('\n', '\\n')
            _debug_log(raw, t, f'"{content_preview}" pos=({position[0]:.1f},{position[1]:.1f}) h={height:.2f}')
            if debug:
                print(f"{prefix}[{t}] handle={_get_handle(raw)} "
                      f"\"{content_preview}\" pos=({position[0]:.1f},{position[1]:.1f}) h={height:.2f}")
            return DxfText(entity_type=t, layer=layer, color=color,
                           text=text, position=position,
                           height=float(height), rotation=rotation,
                           width_factor=width_factor,
                           horizontal_alignment=halign,
                           vertical_alignment=valign,
                           attachment_point=attachment_point,
                           style=str(getattr(raw.dxf, "style", "Standard")))

        elif t == "INSERT":
            _debug_log(raw, "INSERT (expand)", "")
            if debug:
                print(f"{prefix}[INSERT] handle={_get_handle(raw)} "
                      f"layer={layer} → virtual_entities()")
            for attrib in getattr(raw, "attribs", ()):
                sub = _parse_entity(attrib, depth + 1, default_text_height)
                if sub is not None:
                    _dispatch(sub)
                    result.insert_attribute_count += 1

            try:
                for virt in raw.virtual_entities():
                    sub = _parse_entity(virt, depth + 1, default_text_height)
                    if sub is not None:
                        sub.layer = layer
                        sub.color = color
                        _dispatch(sub)
            except Exception as e:
                if debug:
                    print(f"{prefix}  ✗ INSERT expand failed: {e}")
            return None

        elif t == "DIMENSION":
            result.source_dimension_count += 1
            _debug_log(raw, "DIMENSION (expand)", "")
            if debug:
                print(f"{prefix}[DIMENSION] handle={_get_handle(raw)} → virtual_entities()")

            dim_height = _dimension_text_height(raw)
            try:
                virtual_entities = list(raw.virtual_entities())
            except Exception:
                virtual_entities = []

            text_count_before = len(result.texts)
            try:
                for virt in virtual_entities:
                    sub = _parse_entity(virt, depth + 1, dim_height)
                    if sub is not None:
                        _dispatch(sub)
            except Exception as e:
                if debug:
                    print(f"{prefix}  ✗ DIMENSION expand failed: {e}")

            if len(result.texts) == text_count_before:
                try:
                    dim_dxf = raw.dxf
                    cleaned = _dimension_fallback_text(raw)
                    if cleaned:
                        tp = (0.0, 0.0, 0.0)
                        if hasattr(dim_dxf, 'text_midpoint') and dim_dxf.text_midpoint:
                            tp = (dim_dxf.text_midpoint.x, dim_dxf.text_midpoint.y, 0.0)
                        result.dimensions.append(DxfDimension(
                            entity_type="DIMENSION", layer=layer, color=color,
                            text=cleaned, text_position=tp, text_height=dim_height,
                        ))
                        _add(*tp)
                except Exception:
                    pass
            return None

        elif t in ("SOLID", "TRACE"):
            pts = []
            for attr_name in ('vtx0', 'vtx1', 'vtx2', 'vtx3'):
                if hasattr(raw.dxf, attr_name):
                    v = getattr(raw.dxf, attr_name)
                    pts.append((v.x, v.y))
                    _add(v.x, v.y)
            if pts:
                _debug_log(raw, "SOLID", f"{len(pts)}pts")
                return DxfPolyline(entity_type=t, layer=layer, color=color,
                                   points=pts, closed=True)
            return None

        elif t == "POINT":
            p = raw.dxf.location
            _add(p.x, p.y, p.z)
            _debug_log(raw, "POINT", f"({p.x:.1f},{p.y:.1f})")
            return DxfCircle(entity_type="POINT", layer=layer, color=color,
                             center=(p.x, p.y, p.z), radius=1.0)

        elif t == "HATCH":
            try:
                for path in raw.paths:
                    if hasattr(path, 'vertices'):
                        for v in path.vertices:
                            _add(v[0], v[1])
            except Exception:
                pass
            return None

        elif t == "LEADER":
            if debug:
                print(f"{prefix}[LEADER] handle={_get_handle(raw)} → virtual_entities()")
            try:
                for virt in raw.virtual_entities():
                    sub = _parse_entity(virt, depth + 1)
                    if sub is not None:
                        _dispatch(sub)
            except Exception:
                pass
            return None

        elif t == "MLINE":
            if debug:
                print(f"{prefix}[MLINE] handle={_get_handle(raw)} → virtual_entities()")
            try:
                for virt in raw.virtual_entities():
                    sub = _parse_entity(virt, depth + 1)
                    if sub is not None:
                        _dispatch(sub)
            except Exception:
                pass
            return None

        elif t in ("IMAGE", "WIPEOUT", "XLINE", "RAY"):
            return None

        else:
            if debug:
                print(f"{prefix}[UNKNOWN] {t} handle={_get_handle(raw)}")
            return None

    # ============================================================
    # 主循环
    # ============================================================
    for entity in msp:
        parsed = _parse_entity(entity)
        if parsed is not None:
            _dispatch(parsed)

    # ============================================================
    # Debug 报告
    # ============================================================
    if debug:
        print(f"\n{'='*60}")
        print(f"📊 Debug 报告")
        print(f"{'='*60}")
        print(f"\n--- 原始 DXF 实体类型统计 (含 virtual_entities 展开的子实体) ---")
        for t, c in sorted(debug_type_count.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
        print(f"\n--- 最终 DxfDocument 实体数量 ---")
        print(f"  all_entities: {len(result.all_entities)}")
        print(f"  lines:  {len(result.lines)}")
        print(f"  polys:  {len(result.polylines)}")
        print(f"  circles:{len(result.circles)}")
        print(f"  arcs:   {len(result.arcs)}")
        print(f"  splines:{len(result.splines)}")
        print(f"  texts:  {len(result.texts)}")
        print(f"  dims:   {len(result.dimensions)}")
        print(f"  blocks: {len(result.block_names)}")
        print(f"\n--- 文字实体详情 (前30条) ---")
        for line in debug_text_log[:30]:
            print(f"  {line}")
        if len(debug_text_log) > 30:
            print(f"  ... 共 {len(debug_text_log)} 条文字记录")
        repeats = {h: c for h, c in debug_handles_seen.items() if c > 1}
        if repeats:
            print(f"\n⚠️  重复 handle ({len(repeats)} 个):")
            for h, c in sorted(repeats.items(), key=lambda x: -x[1])[:10]:
                print(f"  handle={h} 出现 {c} 次")
        else:
            print(f"\n✅ 无重复 handle")
        if debug_text_log:
            print(f"\n--- 文字位置采样 (逐条) ---")
            for line in debug_text_log:
                print(f"  {line}")

    # 包围盒
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        zs = [p[2] for p in all_pts]
        result.extents = {
            "xmin": min(xs), "xmax": max(xs),
            "ymin": min(ys), "ymax": max(ys),
            "zmin": min(zs), "zmax": max(zs),
        }

    logger.info(
        f"DXF 读取成功: {file_path} → "
        f"{result.total_entities} 个实体, "
        f"{len(result.layers)} 图层, "
        f"{len(result.block_names)} 块 "
        f"(texts={len(result.texts)}, dims={len(result.dimensions)})"
    )
    return result


# ============================================================
# 颜色映射 (ezdxf color index → RGB / 图层 → RGB)
# ============================================================

# ACI (AutoCAD Color Index) 前 9 种标准颜色 + 特殊索引
ACI_COLORS = {
    0:  (0.0, 0.0, 0.0),       # ByBlock (默认黑色)
    1:  (1.0, 0.0, 0.0),       # Red
    2:  (1.0, 1.0, 0.0),       # Yellow
    3:  (0.0, 1.0, 0.0),       # Green
    4:  (0.0, 1.0, 1.0),       # Cyan
    5:  (0.0, 0.0, 1.0),       # Blue
    6:  (1.0, 0.0, 1.0),       # Magenta
    7:  (1.0, 1.0, 1.0),       # White (显示为黑色在白色背景上)
    8:  (0.3, 0.3, 0.3),       # Dark Gray
    9:  (0.5, 0.5, 0.5),       # Light Gray
    256:(0.0, 0.0, 0.0),       # ByLayer
}

# ------------------------------------------------------------------
# 图层名 → 标准颜色映射
# 参考: GB/T 14665-2012 (机械工程 CAD 制图规则) 及 AutoCAD 行业惯例
# ------------------------------------------------------------------
_LAYER_PATTERN_MAP = [
    (["DIM", "尺寸", "标注", "PUB_DIM", "BTL_DIM"],
     (0.10, 0.70, 0.10), "尺寸标注"),
    (["CEN", "CENTER", "中心", "轴线", "AXIS", "SYM"],
     (0.85, 0.10, 0.10), "中心线/轴线"),
    (["HID", "HIDDEN", "虚线", "隐藏", "DASH"],
     (0.80, 0.65, 0.00), "隐藏线/虚线"),
    (["TEXT", "文字", "注释", "NOTE", "LABEL", "MTEXT", "MARK"],
     (0.00, 0.55, 0.55), "文字注释"),
    (["HATCH", "剖面", "填充", "阴影", "PATTERN", "BHATCH"],
     (0.50, 0.50, 0.50), "剖面线/填充"),
    (["BORDER", "TITLE", "图框", "标题", "FRAME", "SHEET"],
     (0.15, 0.15, 0.15), "图框/标题栏"),
    (["SECTION", "截面", "剖切", "CUT"],
     (0.70, 0.10, 0.70), "剖切符号"),
    (["WALL", "墙", "WLL"],
     (0.40, 0.30, 0.20), "墙体"),
    (["DOOR", "门", "DR"],
     (0.10, 0.15, 0.85), "门"),
    (["WINDOW", "窗", "WND", "WD"],
     (0.00, 0.60, 0.60), "窗"),
    (["OUTLINE", "CONTOUR", "轮廓", "粗实线", "VISIBLE", "0"],
     (0.08, 0.08, 0.08), "轮廓线/粗实线"),
    (["DEFPOINTS", "REF", "参考", "辅助", "GUIDE", "CONSTRUCTION"],
     (0.65, 0.65, 0.65), "参考线/辅助线"),
    (["ELEC", "电气", "WIRE", "CABLE", "POWER", "SIGNAL"],
     (0.10, 0.20, 0.80), "电气线路"),
    (["PIPE", "管", "PLUMB", "WATER", "GAS"],
     (0.00, 0.50, 0.70), "管道/给排水"),
    (["STEEL", "钢", "BEAM", "COLUMN", "TRUSS"],
     (0.70, 0.40, 0.10), "钢结构"),
]

# 未知图层的稳定调色板 — 24 种高辨识度颜色
_STABLE_PALETTE = [
    (0.12, 0.35, 0.65),   # 蓝
    (0.65, 0.25, 0.15),   # 棕红
    (0.15, 0.60, 0.35),   # 墨绿
    (0.70, 0.55, 0.05),   # 暗金
    (0.45, 0.20, 0.60),   # 紫
    (0.20, 0.50, 0.55),   # 青
    (0.60, 0.35, 0.15),   # 橙棕
    (0.25, 0.40, 0.25),   # 橄榄绿
    (0.55, 0.20, 0.35),   # 玫红
    (0.15, 0.45, 0.70),   # 天蓝
    (0.55, 0.50, 0.05),   # 橄榄
    (0.35, 0.25, 0.50),   # 暗紫
    (0.50, 0.15, 0.20),   # 深红
    (0.15, 0.55, 0.55),   # 深青
    (0.60, 0.45, 0.15),   # 古铜
    (0.25, 0.30, 0.55),   # 灰蓝
    (0.50, 0.35, 0.05),   # 暗黄
    (0.30, 0.20, 0.40),   # 深紫
    (0.20, 0.55, 0.25),   # 翠绿
    (0.65, 0.30, 0.30),   # 暗红
    (0.20, 0.35, 0.45),   # 灰青
    (0.45, 0.35, 0.25),   # 灰棕
    (0.30, 0.45, 0.30),   # 灰绿
    (0.40, 0.25, 0.45),   # 灰紫
]


def _smart_layer_color(layer_name: str) -> Tuple[float, float, float]:
    """
    根据图层名称智能分配符合 CAD 规范的颜色.

    优先匹配常见图层命名模式 (中/英文, 不区分大小写),
    未知图层使用基于名称 hash 的稳定调色板.

    参考: GB/T 14665-2012 机械工程 CAD 制图规则.
    """
    name = layer_name.strip().upper()

    for patterns, color, _desc in _LAYER_PATTERN_MAP:
        for pat in patterns:
            if pat.upper() in name:
                return color

    hash_val = zlib.crc32(layer_name.encode("utf-8"))
    idx = hash_val % len(_STABLE_PALETTE)
    return _STABLE_PALETTE[idx]


def _layer_color(layer_name: str) -> Tuple[float, float, float]:
    """兼容旧接口 — 内部委托给 _smart_layer_color()."""
    return _smart_layer_color(layer_name)


def _aci_to_rgb(color_index: int) -> Tuple[float, float, float]:
    """
    将 ACI 颜色索引转为 RGB (0-1).

    ACI 索引结构:
      0     — ByBlock
      1-9   — 标准色
      10-249— HSL 色谱 (色调以 10 为步进)
      250-255— 灰度级
      256   — ByLayer
    """
    WHITE_BG_REMAP = {
        0:   (0.08, 0.08, 0.08),
        7:   (0.15, 0.15, 0.15),
        8:   (0.28, 0.28, 0.28),
        9:   (0.45, 0.45, 0.45),
        256: (0.08, 0.08, 0.08),
    }
    if color_index in WHITE_BG_REMAP:
        return WHITE_BG_REMAP[color_index]

    ACI_VISIBLE = {
        1:  (0.85, 0.10, 0.10),
        2:  (0.80, 0.65, 0.00),
        3:  (0.10, 0.75, 0.10),
        4:  (0.00, 0.60, 0.60),
        5:  (0.10, 0.10, 0.85),
        6:  (0.75, 0.10, 0.75),
    }
    if color_index in ACI_VISIBLE:
        return ACI_VISIBLE[color_index]

    if 10 <= color_index <= 249:
        import colorsys
        group = (color_index - 10) // 10
        step = (color_index - 10) % 10

        hue = (group * 15) / 360.0
        t = step / 9.0
        lightness = 0.30 + t * 0.20
        saturation = 0.90 - t * 0.30
        return colorsys.hls_to_rgb(hue, lightness, saturation)

    GRAYSCALE = {
        250: (0.20, 0.20, 0.20),
        251: (0.30, 0.30, 0.30),
        252: (0.40, 0.40, 0.40),
        253: (0.50, 0.50, 0.50),
        254: (0.60, 0.60, 0.60),
        255: (0.70, 0.70, 0.70),
    }
    if color_index in GRAYSCALE:
        return GRAYSCALE[color_index]

    return (0.08, 0.08, 0.08)


def _get_color(entity, use_layer_colors: bool) -> Tuple[float, float, float]:
    """
    获取实体颜色, 遵循三级优先级:

    1. 实体自身的 ACI 颜色 (非 ByLayer 256 且非 ByBlock 0)
    2. 智能图层颜色 (识别图层名模式, 符合 CAD 规范)
    3. 稳定调色板后备

    当 use_layer_colors=True 时, 跳过优先级 1, 直接使用图层色.
    """
    if use_layer_colors:
        return _smart_layer_color(entity.layer)

    if entity.color not in (0, 256):
        return _aci_to_rgb(entity.color)

    return _smart_layer_color(entity.layer)


# ============================================================
# DXF 实体 → VTK Actor 转换
# ============================================================

def dxf_to_actors(
    doc: DxfDocument,
    line_width: float = 1.0,
    circle_resolution: int = 64,
    text_visible: bool = True,
    use_layer_colors: bool = False,
    text_scale: float = 1.0,
) -> List[vtk.vtkActor]:
    """
    将 DXF 文档中的所有实体转换为 VTK Actor 列表.

    Parameters
    ----------
    doc : DxfDocument
        DXF 文档.
    line_width : float
        线条宽度.
    circle_resolution : int
        圆/弧的采样分段数.
    text_visible : bool
        是否显示文字.
    use_layer_colors : bool
        是否按图层着色.
    text_scale : float
        文字整体缩放倍率，1.0 表示严格使用 DXF 实体字高.

    Returns
    -------
    list[vtk.vtkActor]
        VTK Actor 列表.
    """
    actors = []

    font_scale = max(float(text_scale), 0.01)

    for entity in doc.lines:
        color = _get_color(entity, use_layer_colors)
        actors.append(_line_to_actor(entity, color, line_width))

    for entity in doc.polylines:
        color = _get_color(entity, use_layer_colors)
        actors.append(_polyline_to_actor(entity, color, line_width))

    for entity in doc.circles:
        color = _get_color(entity, use_layer_colors)
        segs = max(32, int(entity.radius * 2))
        actors.append(_circle_to_actor(entity, color, line_width, segs))

    for entity in doc.arcs:
        color = _get_color(entity, use_layer_colors)
        arc_span = abs(entity.end_angle - entity.start_angle)
        segs = max(16, int(entity.radius * arc_span / 90.0))
        actors.append(_arc_to_actor(entity, color, line_width, segs))

    for entity in doc.ellipses:
        color = _get_color(entity, use_layer_colors)
        actors.append(_ellipse_to_actor(entity, color, line_width, circle_resolution))

    for entity in doc.splines:
        color = _get_color(entity, use_layer_colors)
        actors.append(_spline_to_actor(entity, color, line_width))

    if text_visible:
        for entity in doc.texts:
            color = _get_color(entity, use_layer_colors)
            actors.append(_text_to_actor(entity, color, font_scale))

    if text_visible:
        for dim in doc.dimensions:
            dim_actors = _dimension_to_actors(dim, line_width, font_scale)
            actors.extend(dim_actors)

    return actors


def _compute_font_scale(doc: DxfDocument) -> float:
    """兼容旧调用：DXF 文字使用实体字高，不再依赖全图包围盒."""
    _ = doc
    return 1.0


# ------------------------------------------------------------------
# 各个实体类型的 VTK 转换函数
# ------------------------------------------------------------------

def _line_to_actor(
    line: DxfLine, color: Tuple[float, float, float], width: float
) -> vtk.vtkActor:
    """DXF LINE → VTK line actor"""
    source = vtk.vtkLineSource()
    source.SetPoint1(line.start)
    source.SetPoint2(line.end)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def _polyline_to_actor(
    pline: DxfPolyline, color: Tuple[float, float, float], width: float
) -> vtk.vtkActor:
    """DXF LWPOLYLINE → VTK polyline actor"""
    points = vtk.vtkPoints()
    for pt in pline.points:
        points.InsertNextPoint(pt[0], pt[1], 0.0)

    if pline.closed and len(pline.points) > 0:
        points.InsertNextPoint(pline.points[0][0], pline.points[0][1], 0.0)

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
    for i in range(points.GetNumberOfPoints()):
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def _circle_to_actor(
    circle: DxfCircle, color: Tuple[float, float, float],
    width: float, resolution: int,
) -> vtk.vtkActor:
    """DXF CIRCLE → VTK polyline (正多边形逼近)"""
    points = vtk.vtkPoints()
    cx, cy, cz = circle.center
    r = circle.radius

    for i in range(resolution):
        angle = 2.0 * math.pi * i / resolution
        points.InsertNextPoint(cx + r * math.cos(angle), cy + r * math.sin(angle), cz)

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(resolution + 1)
    for i in range(resolution):
        polyline.GetPointIds().SetId(i, i)
    polyline.GetPointIds().SetId(resolution, 0)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def _arc_to_actor(
    arc: DxfArc, color: Tuple[float, float, float],
    width: float, resolution: int,
) -> vtk.vtkActor:
    """DXF ARC → VTK polyline"""
    points = vtk.vtkPoints()
    cx, cy, cz = arc.center
    r = arc.radius
    start_rad = math.radians(arc.start_angle)
    end_rad = math.radians(arc.end_angle)

    if end_rad <= start_rad:
        end_rad += 2.0 * math.pi

    n_pts = max(2, resolution)
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(n_pts)
    for i in range(n_pts):
        t = i / (n_pts - 1)
        angle = start_rad + t * (end_rad - start_rad)
        points.InsertNextPoint(cx + r * math.cos(angle), cy + r * math.sin(angle), cz)
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def _ellipse_to_actor(
    ellipse: DxfEllipse, color: Tuple[float, float, float],
    width: float, resolution: int,
) -> vtk.vtkActor:
    """DXF ELLIPSE → VTK polyline (参数方程采样)"""
    points = vtk.vtkPoints()
    cx, cy, cz = ellipse.center
    mx, my, mz = ellipse.major_axis
    a = math.sqrt(mx*mx + my*my + mz*mz)
    b = a * ellipse.ratio
    rot_angle = math.atan2(my, mx)

    n_pts = max(2, resolution)
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(n_pts + 1)
    for i in range(n_pts):
        angle = 2.0 * math.pi * i / n_pts
        ex = a * math.cos(angle)
        ey = b * math.sin(angle)
        rx = ex * math.cos(rot_angle) - ey * math.sin(rot_angle)
        ry = ex * math.sin(rot_angle) + ey * math.cos(rot_angle)
        points.InsertNextPoint(cx + rx, cy + ry, cz)
        polyline.GetPointIds().SetId(i, i)
    polyline.GetPointIds().SetId(n_pts, 0)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def _spline_to_actor(
    spline: DxfSpline, color: Tuple[float, float, float], width: float
) -> vtk.vtkActor:
    """DXF SPLINE → VTK polyline (BSpline.approximate 采样后的点列)"""
    points = vtk.vtkPoints()
    for pt in spline.points:
        points.InsertNextPoint(pt)

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
    for i in range(points.GetNumberOfPoints()):
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


# ------------------------------------------------------------------
# 文字渲染
# ------------------------------------------------------------------

_CJK_FONT_FILE = None


def _get_cjk_font() -> Optional[str]:
    """查找系统上可用的 CJK 字体文件."""
    global _CJK_FONT_FILE
    if _CJK_FONT_FILE is not None:
        return _CJK_FONT_FILE

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/Alibaba-PuHuiTi-Regular.otf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/wps-office/HTYYXSTJ.ttf",
        "/usr/share/fonts/wps-office/HYZYB5.ttf",
    ]

    for fp in candidates:
        if os.path.exists(fp):
            _CJK_FONT_FILE = fp
            return fp
    return None


def _text_to_actor(
    text: DxfText, color: Tuple[float, float, float],
    font_scale: float = 1.0,
) -> "vtk.vtkTextActor3D":
    """
    DXF TEXT → vtkTextActor3D (3D 世界空间文字, 支持中文).

    vtkTextActor3D 渲染真实 3D 几何文字，并通过 Actor Scale 将固定
    高分辨率字形映射到 DXF 世界坐标字高。
    """
    actor = vtk.vtkTextActor3D()
    actor.SetInput(text.text)

    prop = actor.GetTextProperty()
    font_path = _get_cjk_font()
    if font_path:
        prop.SetFontFile(font_path)
        prop.SetFontFamily(vtk.VTK_FONT_FILE)

    reference_font_size = 100
    prop.SetFontSize(reference_font_size)
    prop.SetColor(color)
    prop.BoldOff()
    prop.ItalicOff()

    attachment = int(text.attachment_point)
    if attachment:
        horizontal = (attachment - 1) % 3
        vertical = (attachment - 1) // 3
        if horizontal == 1:
            prop.SetJustificationToCentered()
        elif horizontal == 2:
            prop.SetJustificationToRight()
        else:
            prop.SetJustificationToLeft()
        if vertical == 0:
            prop.SetVerticalJustificationToTop()
        elif vertical == 1:
            prop.SetVerticalJustificationToCentered()
        else:
            prop.SetVerticalJustificationToBottom()
    else:
        if text.horizontal_alignment in (1, 3, 4):
            prop.SetJustificationToCentered()
        elif text.horizontal_alignment in (2, 5):
            prop.SetJustificationToRight()
        else:
            prop.SetJustificationToLeft()
        if text.vertical_alignment == 3:
            prop.SetVerticalJustificationToTop()
        elif text.vertical_alignment == 2:
            prop.SetVerticalJustificationToCentered()
        else:
            prop.SetVerticalJustificationToBottom()

    world_scale = max(float(text.height), 0.01) * font_scale / reference_font_size
    width_factor = max(0.05, min(float(text.width_factor), 20.0))
    actor.SetScale(world_scale * width_factor, world_scale, world_scale)

    x, y, z = text.position
    actor.SetPosition(x, y, z + 0.01)
    actor.SetOrientation(0.0, 0.0, float(text.rotation))
    actor.PickableOff()

    return actor


# ------------------------------------------------------------------
# 尺寸标注
# ------------------------------------------------------------------

def _dimension_to_actors(
    dim: DxfDimension, line_width: float = 1.0,
    font_scale: float = 1.0,
) -> List[vtk.vtkActor]:
    """
    将 DXF 尺寸标注转换为 VTK Actor 列表.

    当前仅渲染标注文字标签, 不绘制几何线
    (DXF 定义点含义因标注类型而异, 简化连线容易错乱).
    """
    actors: List[vtk.vtkActor] = []

    if dim.text and dim.text.strip():
        fake_text = DxfText(
            text=dim.text.strip(),
            position=dim.text_position,
            height=dim.text_height,
            layer=dim.layer,
            color=dim.color,
        )
        dim_color = _get_color(dim, use_layer_colors=False)
        actors.append(_text_to_actor(fake_text, dim_color, font_scale))

    return actors
