"""
STEP 文件读取与 VTK 数据转换.

读取 STEP 文件 → OpenCascade TopoDS_Shape,
并提供 Shape → vtkPolyData / vtkActor 的转换功能.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import vtk

logger = logging.getLogger(__name__)


# ============================================================
# STEP 读取 (基于 OCP / OpenCascade)
# ============================================================

class StepReadError(Exception):
    """STEP 文件读取异常"""
    pass


def read_step(filepath: str) -> "TopoDS_Shape":
    """
    读取 STEP 文件，返回 OpenCascade TopoDS_Shape.

    Parameters
    ----------
    filepath : str
        STEP 文件路径 (.step 或 .stp).

    Returns
    -------
    TopoDS_Shape
        模型的拓扑形状对象.

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出.
    StepReadError
        读取或解析失败时抛出.
    """
    from OCP.STEPControl import STEPControl_Reader

    file_path = Path(filepath).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"STEP 文件不存在: {filepath}")

    reader = STEPControl_Reader()
    read_status = reader.ReadFile(str(file_path))

    if read_status != 1:
        raise StepReadError(
            f"STEP 文件读取失败! 文件: {file_path}, 状态码: {read_status}"
        )

    reader.TransferRoots()
    n_roots = reader.NbRootsForTransfer()
    n_shapes = reader.NbShapes()
    logger.info(f"STEP 读取: {n_roots} 根节点, {n_shapes} 形状")

    if n_shapes == 0:
        raise StepReadError("STEP 文件中未找到任何形状")

    shape = reader.OneShape()
    if shape.IsNull():
        raise StepReadError("提取的形状为空")

    logger.info(f"STEP 读取成功: {file_path}")
    return shape


def get_shape_info(shape: "TopoDS_Shape") -> Dict[str, Any]:
    """
    提取 TopoDS_Shape 的拓扑统计信息.

    Parameters
    ----------
    shape : TopoDS_Shape
        OpenCascade 拓扑形状.

    Returns
    -------
    dict
        包含 solid_count, face_count, edge_count, vertex_count 等信息.
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import (
        TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
        TopAbs_SHELL, TopAbs_WIRE, TopAbs_COMPOUND,
    )
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType

    def _count(shape, shape_type) -> int:
        exp = TopExp_Explorer()
        exp.Init(shape, shape_type)
        count = 0
        while exp.More():
            count += 1
            exp.Next()
        return count

    # 拓扑统计
    stats = {
        "solid_count": _count(shape, TopAbs_SOLID),
        "shell_count": _count(shape, TopAbs_SHELL),
        "face_count": _count(shape, TopAbs_FACE),
        "wire_count": _count(shape, TopAbs_WIRE),
        "edge_count": _count(shape, TopAbs_EDGE),
        "vertex_count": _count(shape, TopAbs_VERTEX),
    }

    if stats["solid_count"] == 0:
        stats["solid_count"] = _count(shape, TopAbs_COMPOUND)
        if stats["solid_count"] > 0:
            stats["contains_compound"] = True

    # 面类型分类
    face_types = {}
    exp = TopExp_Explorer()
    exp.Init(shape, TopAbs_FACE)
    while exp.More():
        face_shape = exp.Current()
        face = TopoDS.Face_s(face_shape)
        adaptor = BRepAdaptor_Surface(face)
        surf_type = adaptor.GetType()
        type_name = _surface_type_name(surf_type)
        face_types[type_name] = face_types.get(type_name, 0) + 1
        exp.Next()
    stats["face_types"] = face_types

    # 包围盒
    bbox = _compute_bbox(shape)
    stats["bbox"] = bbox

    return stats


def _surface_type_name(surf_type) -> str:
    """将 GeomAbs_SurfaceType 枚举转为可读名称"""
    from OCP.GeomAbs import GeomAbs_SurfaceType

    names = {
        GeomAbs_SurfaceType.GeomAbs_Plane: "Plane",
        GeomAbs_SurfaceType.GeomAbs_Cylinder: "Cylinder",
        GeomAbs_SurfaceType.GeomAbs_Cone: "Cone",
        GeomAbs_SurfaceType.GeomAbs_Sphere: "Sphere",
        GeomAbs_SurfaceType.GeomAbs_Torus: "Torus",
        GeomAbs_SurfaceType.GeomAbs_BezierSurface: "Bezier",
        GeomAbs_SurfaceType.GeomAbs_BSplineSurface: "BSpline",
        GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion: "Extrusion",
        GeomAbs_SurfaceType.GeomAbs_OffsetSurface: "Offset",
        GeomAbs_SurfaceType.GeomAbs_OtherSurface: "Other",
    }
    return names.get(surf_type, f"Unknown({surf_type})")


def _compute_bbox(shape: "TopoDS_Shape") -> Dict[str, float]:
    """计算形状的包围盒"""
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return {
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
    }


# ============================================================
# STEP Shape → VTK 数据转换
# ============================================================

def step_to_vtk_polydata(
    shape: "TopoDS_Shape",
    deflection: float = 0.1,
) -> vtk.vtkPolyData:
    """
    将 OpenCascade TopoDS_Shape 转换为 vtkPolyData (三角网格).

    使用 BRepMesh_IncrementalMesh 进行三角剖分，
    逐面提取三角形构建 vtkPolyData，并计算顶点法线.

    Parameters
    ----------
    shape : TopoDS_Shape
        OpenCascade 拓扑形状.
    deflection : float
        三角剖分的线性挠度 (越小越精细).

    Returns
    -------
    vtk.vtkPolyData
        包含三角网格和顶点法线的 VTK 多边形数据.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Pnt, gp_XYZ
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    # 1. 三角剖分
    mesh = BRepMesh_IncrementalMesh(shape, deflection)
    mesh.Perform()

    # 2. 逐面提取三角形
    all_points: List[Tuple[float, float, float]] = []
    all_triangles: List[Tuple[int, int, int]] = []
    vertex_offset = 0

    exp = TopExp_Explorer()
    exp.Init(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)

        if triangulation is not None:
            transformation = location.Transformation()
            n_vertices = triangulation.NbNodes()
            for i in range(1, n_vertices + 1):
                pnt = triangulation.Node(i).Transformed(transformation)
                all_points.append((pnt.X(), pnt.Y(), pnt.Z()))

            orientation = face.Orientation()
            n_triangles = triangulation.NbTriangles()
            for i in range(1, n_triangles + 1):
                tri = triangulation.Triangle(i)
                if orientation == TopAbs_FORWARD:
                    i1, i2, i3 = tri.Get()
                else:
                    i1, i3, i2 = tri.Get()
                all_triangles.append((
                    i1 - 1 + vertex_offset,
                    i2 - 1 + vertex_offset,
                    i3 - 1 + vertex_offset,
                ))

            vertex_offset += n_vertices

        exp.Next()

    if not all_points:
        raise RuntimeError("三角剖分后未提取到任何顶点")

    # 3. 构建 vtkPolyData
    vtk_points = vtk.vtkPoints()
    for pt in all_points:
        vtk_points.InsertNextPoint(pt)

    vtk_cells = vtk.vtkCellArray()
    for tri in all_triangles:
        vtk_cells.InsertNextCell(3)
        vtk_cells.InsertCellPoint(tri[0])
        vtk_cells.InsertCellPoint(tri[1])
        vtk_cells.InsertCellPoint(tri[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    # 4. 计算法线以获得平滑渲染
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def step_to_actor(
    shape: "TopoDS_Shape",
    color: Tuple[float, float, float] = (0.7, 0.7, 0.75),
    alpha: float = 1.0,
    wireframe: bool = False,
    deflection: float = 0.1,
) -> vtk.vtkActor:
    """
    将 OpenCascade TopoDS_Shape 转换为 VTK Actor.

    内部调用 step_to_vtk_polydata() 获取网格数据，
    然后创建 vtkMapper + vtkActor.

    Parameters
    ----------
    shape : TopoDS_Shape
        OpenCascade 拓扑形状.
    color : tuple
        RGB 颜色 (0-1).
    alpha : float
        透明度 (0-1).
    wireframe : bool
        是否显示线框模式.
    deflection : float
        三角剖分的线性挠度 (越小越精细).

    Returns
    -------
    vtk.vtkActor
        可添加到 VTK 渲染器的 Actor.
    """
    polydata = step_to_vtk_polydata(shape, deflection)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(alpha)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetInterpolationToPhong()

    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    return actor


def step_to_wireframe_actor(
    shape: "TopoDS_Shape",
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    deflection: float = 0.1,
) -> vtk.vtkActor:
    """创建线框模式 Actor (带轮廓边缘)."""
    actor = step_to_actor(shape, color=color, deflection=deflection)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(1.0)
    return actor


def step_to_edges_actor(
    shape: "TopoDS_Shape",
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    line_width: float = 1.5,
) -> vtk.vtkActor:
    """
    提取形状的所有边并渲染为可见的线段.

    适用于在实体模型上叠加边线显示.
    """
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_UniformAbscissa

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    point_id = 0

    exp = TopExp_Explorer()
    exp.Init(shape, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        curve_adaptor = BRepAdaptor_Curve(edge)
        first = curve_adaptor.FirstParameter()
        last = curve_adaptor.LastParameter()

        n_samples = 20
        try:
            sampler = GCPnts_UniformAbscissa(curve_adaptor, n_samples + 1, first, last)
            if sampler.IsDone():
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(sampler.NbPoints())
                for i in range(1, sampler.NbPoints() + 1):
                    pnt = sampler.Parameter(i)
                    pt = curve_adaptor.Value(pnt)
                    points.InsertNextPoint(pt.X(), pt.Y(), pt.Z())
                    line.GetPointIds().SetId(i - 1, point_id)
                    point_id += 1
                lines.InsertNextCell(line)
        except Exception:
            pass

        exp.Next()

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(line_width)

    return actor


# 兼容旧名称
shape_to_actor = step_to_actor
shape_to_wireframe_actor = step_to_wireframe_actor
shape_to_edges_actor = step_to_edges_actor
