"""
io — CAD 文件读取与 VTK 数据转换.

Public API:
  STEP:
    - read_step(filepath)           → TopoDS_Shape
    - get_shape_info(shape)         → dict
    - step_to_vtk_polydata(shape)   → vtkPolyData
    - step_to_actor(shape)          → vtkActor
    - step_to_wireframe_actor(shape)→ vtkActor
    - step_to_edges_actor(shape)    → vtkActor

  DXF:
    - read_dxf(filepath)            → DxfDocument
    - dxf_to_actors(doc)            → list[vtkActor]

  Data Models:
    DxfDocument, DxfEntity, DxfLine, DxfPolyline, DxfCircle,
    DxfArc, DxfEllipse, DxfSpline, DxfText, DxfDimension

  Colors:
    ACI_COLORS, _STABLE_PALETTE
    _smart_layer_color, _aci_to_rgb, _get_color
"""

from .step_io import (
    # STEP 读取
    read_step,
    get_shape_info,
    StepReadError,
    # STEP → VTK
    step_to_vtk_polydata,
    step_to_actor,
    step_to_wireframe_actor,
    step_to_edges_actor,
    # 兼容旧名称
    shape_to_actor,
    shape_to_wireframe_actor,
    shape_to_edges_actor,
)

from .dxf_io import (
    # DXF 数据模型
    DxfDocument,
    DxfEntity,
    DxfLine,
    DxfPolyline,
    DxfCircle,
    DxfArc,
    DxfEllipse,
    DxfSpline,
    DxfText,
    DxfDimension,
    # DXF 读取
    read_dxf,
    DxfReadError,
    # DXF → VTK
    dxf_to_actors,
    # 颜色工具
    ACI_COLORS,
    _STABLE_PALETTE,
    _smart_layer_color,
    _layer_color,
    _aci_to_rgb,
    _get_color,
)

__all__ = [
    # STEP
    "read_step",
    "get_shape_info",
    "StepReadError",
    "step_to_vtk_polydata",
    "step_to_actor",
    "step_to_wireframe_actor",
    "step_to_edges_actor",
    "shape_to_actor",
    "shape_to_wireframe_actor",
    "shape_to_edges_actor",
    # DXF
    "DxfDocument",
    "DxfEntity",
    "DxfLine",
    "DxfPolyline",
    "DxfCircle",
    "DxfArc",
    "DxfEllipse",
    "DxfSpline",
    "DxfText",
    "DxfDimension",
    "read_dxf",
    "DxfReadError",
    "dxf_to_actors",
    # Colors
    "ACI_COLORS",
    "_STABLE_PALETTE",
    "_smart_layer_color",
    "_layer_color",
    "_aci_to_rgb",
    "_get_color",
]
