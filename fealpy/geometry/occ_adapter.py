from typing import Any, Optional, Tuple, List, Dict, Literal, Union, Sequence
from pathlib import Path
import os
from math import pi, sin, cos
from ..decorator import multi_input

from .geometry_kernel_adapter_base import (
    GeometryKernelAdapterBase, ATTRIBUTE_MAPPING,
    FUNCTION_MAPPING, TRANSFORMS_MAPPING,
    hex_to_rgb, parse_color
)

import OCC.Core as OCC
from OCC.Core.TopoDS import (
    TopoDS_Shape, TopoDS_Vertex, TopoDS_Edge,
    TopoDS_Wire, TopoDS_Face, TopoDS_Solid,
    TopoDS_Compound, TopoDS_Shell)
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Ax2, gp_Dir, gp_XYZ, gp_Mat

def _ensure_vertex(input_data: Union[TopoDS_Vertex, Tuple[float, float, float]], class_name: str):
    """
        将输入转换为TopoDS_Vertex

        Parameters
        ----------
        input_data : Union[TopoDS_Vertex, Tuple[float, float, float]]
            输入顶点或坐标
        class_name : str
            调用类名（用于错误提示）

        Returns
        -------
        TopoDS_Vertex
            转换后的顶点对象

        Raises
        ------
        TypeError
            输入类型不合法时抛出
        """
    if isinstance(input_data, TopoDS_Vertex):
        return input_data
    elif isinstance(input_data, (tuple, list)) and len(input_data) == 3:
        # 隐式调用add_point创建顶点
        return OCCAdapter.add_point(*input_data)
    else:
        raise TypeError(
            f"{class_name}: 输入类型需为TopoDS_Vertex或三维坐标 (tuple/list), 实际类型为 {type(input_data)}"
        )


def _is_wire_closed(wire: TopoDS_Wire) -> bool:
    """
    检查线框是否闭合（通过首尾顶点坐标对比）

    Parameters
    ----------
    wire : TopoDS_Wire
        待检查的线框

    Returns
    -------
    bool
        True表示闭合，False表示开放
    """
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool

    # 遍历线框中的顶点
    explorer = TopExp_Explorer(wire, TopAbs_VERTEX)  # 使用 TopAbs_Vertex 枚举
    vertices = []
    while explorer.More():
        current_shape = explorer.Current()
        # 将通用 Shape 转换为具体的 Vertex 类型
        vertex = TopoDS_Vertex(current_shape)  # 直接使用 TopoDS_Vertex 构造函数
        vertices.append(BRep_Tool.Pnt(vertex))
        explorer.Next()

    # 检查首尾顶点是否重合
    if len(vertices) < 2:
        return False
    return vertices[0].Distance(vertices[-1]) < 1e-6


def _is_shell_closed(shell: TopoDS_Shell) -> bool:
    """检查壳体是否闭合"""
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    # 1. 获取自由边（开放边）
    free_edges = ShapeAnalysis_FreeBounds(shell, True).GetOpenWires()  # 注意此处改为 GetOpenWires

    # 2. 检查自由边数量
    explorer = TopExp_Explorer(free_edges, TopAbs_EDGE)
    return not explorer.More()  # 无自由边表示闭合



class OCCAdapter(GeometryKernelAdapterBase, adapter_name="occ"):
    def initialize(self, config=None):
        pass

    def shutdown(self):
        pass

    # ===========================================================
    # entity construct
    # 点、线、曲线
    @staticmethod
    def add_point(x: float, y: float, z: float) -> TopoDS_Vertex:
        """
                创建一个三维点

                Parameters
                ----------
                x : float
                    X坐标值
                y : float
                    Y坐标值
                z : float
                    Z坐标值

                Returns
                -------
                TopoDS_Vertex
                    生成的顶点对象
                """
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        pnt = gp_Pnt(x, y, z)
        return BRepBuilderAPI_MakeVertex(pnt).Vertex()

    @staticmethod
    def add_line(
            start_point: Union[TopoDS_Vertex, Tuple[float, float, float]],
            end_point: Union[TopoDS_Vertex, Tuple[float, float, float]]
    ) -> TopoDS_Edge:
        """
        创建线段，支持顶点对象或坐标输入

        Parameters
        ----------
        start_point : Union[TopoDS_Vertex, Tuple[float, float, float]]
            起点（顶点对象或坐标）
        end_point : Union[TopoDS_Vertex, Tuple[float, float, float]]
            终点（顶点对象或坐标）

        Returns
        -------
        TopoDS_Edge
            生成的边对象
        """
        v_start = _ensure_vertex(start_point, "add_line")
        v_end = _ensure_vertex(end_point, "add_line")

        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        edge_maker = BRepBuilderAPI_MakeEdge(v_start, v_end)
        if not edge_maker.IsDone():
            raise RuntimeError("线段创建失败")
        return edge_maker.Edge()

    @staticmethod
    def add_arc(start: Union[TopoDS_Vertex, Tuple[float, float, float]],
                middle: Union[TopoDS_Vertex, Tuple[float, float, float]],
                end: Union[TopoDS_Vertex, Tuple[float, float, float]]) -> TopoDS_Edge:
        """
        通过三点创建圆弧

        Parameters
        ----------
        start : Union[TopoDS_Vertex, Tuple[float, float, float]]
            起始点
        middle : Union[TopoDS_Vertex, Tuple[float, float, float]]
            中间点
        end : Union[TopoDS_Vertex, Tuple[float, float, float]]
            终止点

        Returns
        -------
        TopoDS_Edge
            圆弧边对象
        """
        from OCC.Core.BRep import BRep_Tool
        if isinstance(start, TopoDS_Vertex):
            p_start = BRep_Tool.Pnt(start)
            p_middle = BRep_Tool.Pnt(middle)
            p_end = BRep_Tool.Pnt(end)
        elif isinstance(start, (tuple, list)) and len(start) == 3:
            p_start = gp_Pnt(*start)
            p_middle = gp_Pnt(*middle)
            p_end = gp_Pnt(*end)

        from OCC.Core.GC import GC_MakeArcOfCircle
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        arc_maker = GC_MakeArcOfCircle(p_start, p_middle, p_end)
        if not arc_maker.IsDone():
            raise RuntimeError("三点圆弧创建失败")
        return BRepBuilderAPI_MakeEdge(arc_maker.Value()).Edge()

    @staticmethod
    def add_arc_center(
            center: Union[TopoDS_Vertex, Tuple[float, float, float]],
            start: Union[TopoDS_Vertex, Tuple[float, float, float]],
            end: Union[TopoDS_Vertex, Tuple[float, float, float]]) -> TopoDS_Edge:
        """
        通过圆心+起点+终点创建圆弧

        Parameters
        ----------
        center : Union[TopoDS_Vertex, Tuple[float, float, float]]
            圆心坐标
        start : Union[TopoDS_Vertex, Tuple[float, float, float]]
            起始点坐标
        end : Union[TopoDS_Vertex, Tuple[float, float, float]]
            终止点坐标

        Returns
        -------
        TopoDS_Edge
            圆弧边对象

        Notes
        -----
        圆弧方向遵循右手定则，默认在XY平面
        """
        from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Circ
        from OCC.Core.GC import GC_MakeArcOfCircle
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCC.Core.BRep import BRep_Tool

        if isinstance(center, TopoDS_Vertex):
            p_center = BRep_Tool.Pnt(center)
            p_start = BRep_Tool.Pnt(start)
            p_end = BRep_Tool.Pnt(end)
        elif isinstance(center, (tuple, list)) and len(center) == 3:
            p_center = gp_Pnt(*center)
            p_start = gp_Pnt(*start)
            p_end = gp_Pnt(*end)

        circle = gp_Circ(gp_Ax2(p_center, gp_Dir(0, 0, 1)), p_start.Distance(p_center))
        arc_maker = GC_MakeArcOfCircle(circle, p_start, p_end, True)
        if not arc_maker.IsDone():
            raise RuntimeError("圆心圆弧创建失败")
        return BRepBuilderAPI_MakeEdge(arc_maker.Value()).Edge()

    @staticmethod
    def add_spline(points: Union[TopoDS_Vertex, List[Tuple[float, float, float]]]) -> TopoDS_Edge:
        """
        创建B样条曲线

        Parameters
        ----------
        points : Union[TopoDS_Vertex, List[Tuple[float, float, float]]]
            控制点坐标列表，格式为[(x1,y1,z1), (x2,y2,z2), ...]

        Returns
        -------
        TopoDS_Edge
            样条曲线边对象

        Examples
        --------
        >>> ctrl_points = [(0,0,0), (2,3,1), (5,4,2), (7,1,3)]
        >>> spline = OccGeometryTools.add_spline(ctrl_points)
        """
        from OCC.Core.TColgp import TColgp_Array1OfPnt
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCC.Core.BRep import BRep_Tool
        arr = TColgp_Array1OfPnt(1, len(points))

        if isinstance(points[0], TopoDS_Vertex):
            for i, v in enumerate(points):
                arr.SetValue(i + 1, BRep_Tool.Pnt(v))
        elif isinstance(points[0], (tuple, list)) and len(points[0]) == 3:
            for i, (x, y, z) in enumerate(points):
                arr.SetValue(i + 1, gp_Pnt(x, y, z))

        curve = GeomAPI_PointsToBSpline(arr).Curve()
        if curve is None:
            raise RuntimeError("样条曲线生成失败")
        return BRepBuilderAPI_MakeEdge(curve).Edge()

    @staticmethod
    def add_curve_loop(edges: list) -> TopoDS_Wire:
        """
                将一组边按顺序连接生成闭合线框（Wire）

                Parameters
                ----------
                edges : List[TopoDS_Edge]
                    边列表，需按顺序首尾相连

                Returns
                -------
                TopoDS_Wire
                    生成的闭合线框

                Raises
                ------
                RuntimeError
                    线框无法闭合或几何无效时抛出

                Examples
                --------
                >>> e1 = add_line(p1, p2)
                >>> e2 = add_line(p2, p3)
                >>> e3 = add_line(p3, p1)
                >>> wire = add_curve_loop([e1, e2, e3])
                """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        # 1. 创建线框
        wire_maker = BRepBuilderAPI_MakeWire()
        for edge in edges:
            wire_maker.Add(edge)

        if not wire_maker.IsDone():
            raise RuntimeError("线框闭合失败：边未按顺序首尾相连")

        wire = wire_maker.Wire()

        # 2. 验证闭合性
        if not _is_wire_closed(wire):
            raise RuntimeError("生成的线框未闭合")

        # 3. 验证几何有效性
        analyzer = BRepCheck_Analyzer(wire)
        if not analyzer.IsValid():
            raise RuntimeError("生成的线框存在几何缺陷")

        return wire

    @staticmethod
    def add_face_loop(faces: List[TopoDS_Face]) -> TopoDS_Shell:
        """
        将一组面组合成闭合壳（Shell）

        Parameters
        ----------
        faces : List[TopoDS_Face]
            面列表，需按拓扑顺序连接

        Returns
        -------
        TopoDS_Shell
            生成的闭合壳

        Raises
        ------
        RuntimeError
            壳无法闭合或几何无效时抛出

        Examples
        --------
        >>> f1 = add_plane_surface(wire1)
        >>> f2 = add_plane_surface(wire2)
        >>> shell = add_face_loop([f1, f2])
        """
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        # 1. 创建空壳
        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)

        # 2. 添加面到壳
        for face in faces:
            builder.Add(shell, face)

        # 3. 验证闭合性
        if not _is_shell_closed(shell):
            raise RuntimeError("生成的壳未闭合")

        # TODO: 验证几何有效性
        # # 4. 验证几何有效性
        # analyzer = BRepCheck_Analyzer(shell)
        # if not analyzer.IsValid():
        #     raise RuntimeError("生成的壳存在几何缺陷")

        return shell

    # 面、体
    @staticmethod
    def add_surface(wire: TopoDS_Wire) -> TopoDS_Face:
        """
        基于闭合线框创建平面面

        Parameters
        ----------
        wire : TopoDS_Wire
            闭合线框（必须为单一线框且闭合）

        Returns
        -------
        TopoDS_Face
            生成的平面面

        Raises
        ------
        ValueError
            输入线框不闭合或非单一线框时抛出
        RuntimeError
            面创建失败时抛出

        Examples
        --------
        >>> rectangle = OccGeometryTools.add_rectangle(0,0,0, 10, 5)
        >>> face = OccGeometryTools.add_plane_surface(rectangle)
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        # 1. 校验线框闭合性
        if not _is_wire_closed(wire):
            raise ValueError("输入线框必须为闭合状态")

        # 2. 创建平面面
        face_maker = BRepBuilderAPI_MakeFace(wire)
        # if not face_maker.IsDone():
        #     raise RuntimeError("平面面创建失败：线框可能不在同一平面或包含无效几何")

        # 3. 验证面的有效性
        analyzer = BRepCheck_Analyzer(face_maker.Face())
        if not analyzer.IsValid():
            raise RuntimeError("生成的平面面几何无效")

        return face_maker.Face()

    @staticmethod
    def add_volume(shell: TopoDS_Shell) -> TopoDS_Solid:
        """
        将闭合壳体转换为实体

        Parameters
        ----------
        shell : TopoDS_Shell
            闭合壳体（必须为完全闭合的壳体）

        Returns
        -------
        TopoDS_Solid
            生成的实体

        Raises
        ------
        ValueError
            壳体未闭合时抛出
        RuntimeError
            实体创建失败时抛出

        Examples
        --------
        >>> # 假设已创建闭合壳体（例如球体）
        >>> sphere_shell = OccGeometryTools.add_sphere(0,0,0,5).Shell()
        >>> solid = OccGeometryTools.add_volume(sphere_shell)
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        # 1. 校验壳体闭合性
        if not _is_shell_closed(shell):
            raise ValueError("输入壳体必须为闭合状态")

        # 2. 创建实体
        solid_maker = BRepBuilderAPI_MakeSolid()
        solid_maker.Add(shell)
        if not solid_maker.IsDone():
            raise RuntimeError("实体创建失败：壳体可能未完全闭合或包含间隙")

        # # 3. 验证实体有效性
        # analyzer = BRepCheck_Analyzer(solid_maker.Solid())
        # if not analyzer.IsValid():
        #     raise RuntimeError("生成的实体几何无效")

        return solid_maker.Solid()

    @staticmethod
    def add_helix(
        radius: float,
        pitch: float,
        height: float,
        start_point: Tuple[float, float, float] = (0, 0, 0),
        axis_direction: Tuple[float, float, float] = (0, 0, 1)
    ) -> TopoDS_Edge:
        """
        创建螺旋线（兼容旧版 OCC）

        Parameters
        ----------
        radius : float
            螺旋半径（必须 > 0）
        pitch : float
            螺距（每圈上升高度，必须 > 0）
        height : float
            总高度（必须 > 0）
        start_point : Tuple[float, float, float], optional
            起始点坐标，默认 (0, 0, 0)
        axis_direction : Tuple[float, float, float], optional
            螺旋轴方向，默认沿Z轴 (0, 0, 1)

        Returns
        -------
        TopoDS_Edge
            螺旋线边对象
        """
        raise NotImplementedError("该方法还未实现。")
        if radius <= 0 or pitch <= 0 or height <= 0:
            raise ValueError("半径、螺距和高度必须为正数")
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

        # 计算螺旋参数
        num_turns = height / pitch
        step = height / (num_turns * 100)  # 采样点密度
        points = []
        axis_dir = gp_Dir(*axis_direction)
        axis_vec = gp_Vec(axis_dir) * step

        # 生成螺旋采样点
        for i in range(int(num_turns * 100) + 1):
            t = i * step
            theta = 2 * pi * t / pitch
            x = radius * cos(theta)
            y = radius * sin(theta)
            z = t
            pnt = gp_Pnt(x, y, z)
            points.append(pnt)

        # 构建样条曲线
        curve = OCCAdapter.add_spline(points)
        return curve



    # ===========================================================
    # bool operate
    @staticmethod
    def boolean_cut(shape1, shape2) -> TopoDS_Shape:
        """
        对两个实体进行切割
        Parameters
        ----------
        shape1: OCC 实体
        shape2: OCC 实体

        Returns
        -------

        """
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        return BRepAlgoAPI_Cut(shape1, shape2).Shape()

    @staticmethod
    def boolean_union(*shapes) -> TopoDS_Shape:
        """
        对两个几何体进行布尔并集操作，返回合并后的形状。
        Parameters
        ----------
        shapes : OCC.TopoDS_Shape 几何体 (支持多个)

        Returns
        -------
        TopoDS_Shape
            合并后的几何体
        """
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        if len(shapes) == 1 and not shapes[0]:
            result = shapes[0]
        elif len(shapes) >= 2:
            for idx, shape in enumerate(shapes):
                if not shape:
                    raise ValueError("几何体不能为空。")
                if idx == 0:
                    result = shape
                    continue
                # 创建布尔操作工具
                result = BRepAlgoAPI_Fuse(result, shape).Shape()


        # 验证结果的有效性
        analyzer = BRepCheck_Analyzer(result)
        if not analyzer.IsValid():
            raise RuntimeError("布尔并集操作失败：生成的几何体无效。")

        return result

    @staticmethod
    def boolean_intersect(
            shape1: TopoDS_Shape,
            shape2: TopoDS_Shape
    ) -> TopoDS_Shape:
        """
        计算两个几何体的布尔交集

        Parameters
        ----------
        shape1 : TopoDS_Shape
            第一个几何体（需为实体）
        shape2 : TopoDS_Shape
            第二个几何体（需为实体）

        Returns
        -------
        TopoDS_Shape
            交集结果

        Raises
        ------
        RuntimeError
            操作失败时抛出异常

        Examples
        --------
        >>> box1 = OccGeometryTools.add_box(0, 0, 0, 10, 10, 10)
        >>> box2 = OccGeometryTools.add_box(5, 5, 0, 10, 10, 10)
        >>> intersection = OccGeometryTools.boolean_intersect(box1, box2)
        """
        from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        if not (shape1.ShapeType() == TopAbs_SOLID and
                shape2.ShapeType() == TopAbs_SOLID):
            raise TypeError("输入必须为实体类型(Solid)")

        common = BRepAlgoAPI_Common(shape1, shape2)
        if not common.IsDone():
            raise RuntimeError("布尔交集操作失败")

        result = common.Shape()
        analyzer = BRepCheck_Analyzer(result)
        if not analyzer.IsValid():
            raise RuntimeError("生成的交集几何体无效")
        return result


    @staticmethod
    def boolean_fragment(
            *shapes: TopoDS_Shape
    ) -> TopoDS_Compound:
        """
        对多个几何体进行非融合切割，保留所有碎片

        Parameters
        ----------
        *shapes : TopoDS_Shape
            待切割的几何体（至少两个）

        Returns
        -------
        TopoDS_Compound
            包含所有碎片的复合体

        Examples
        --------
        >>> box1 = add_box(0, 0, 0, 10, 10, 10)
        >>> box2 = add_box(5, 5, 0, 10, 10, 10)
        >>> fragments = boolean_fragment(box1, box2)
        """
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Splitter
        from OCC.Core.TopTools import TopTools_ListOfShape
        if len(shapes) < 2:
            raise ValueError("至少需要两个几何体进行切割")

        # 使用 TopTools_ListOfShape 来存储多个形状
        arguments = TopTools_ListOfShape()
        tools = TopTools_ListOfShape()

        # 第一个形状作为 base，其余的作为 tools
        arguments.Append(shapes[0])
        for shape in shapes[1:]:
            tools.Append(shape)

        splitter = BRepAlgoAPI_Splitter()
        splitter.SetArguments(arguments)  # 正确传递 TopTools_ListOfShape
        splitter.SetTools(tools)  # 正确传递 TopTools_ListOfShape
        splitter.Build()

        if not splitter.IsDone():
            raise RuntimeError("非融合切割操作失败")

        return splitter.Shape()

    # ===========================================================
    # geometry operate
    @staticmethod
    def translate(
            shape: TopoDS_Shape,
            vec: Union[Tuple[float, float, float], gp_Vec]
    ) -> TopoDS_Shape:
        """
        平移几何体

        Parameters
        ----------
        shape : TopoDS_Shape
            待平移的几何体
        vec : Union[Tuple[float, float, float], gp_Vec]
            平移向量 (x, y, z) 或 gp_Vec 对象

        Returns
        -------
        TopoDS_Shape
            平移后的几何体
        """
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        if isinstance(vec, tuple):
            vec = gp_Vec(*vec)

        trsf = gp_Trsf()
        trsf.SetTranslation(vec)
        return BRepBuilderAPI_Transform(shape, trsf, True).Shape()

    @staticmethod
    def rotate(
            shape: TopoDS_Shape,
            rotation_point: Union[Tuple[float, float, float], gp_Pnt],
            rotation_axis: Union[Tuple[float, float, float], gp_Dir],
            angle: float
    ) -> TopoDS_Shape:
        """
        绕指定轴旋转几何体

        Parameters
        ----------
        shape : TopoDS_Shape
            待旋转的几何体
        rotation_point : Union[Tuple[float, float, float], gp_Pnt]
            旋转中心点
        rotation_axis : Union[Tuple[float, float, float], gp_Dir]
            旋转轴
        angle : float
            旋转角度（弧度）

        Returns
        -------
        TopoDS_Shape
            旋转后的几何体
        """
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

        if isinstance(rotation_point, tuple):
            rotation_point = gp_Pnt(*rotation_point)
        if isinstance(rotation_axis, tuple):
            rotation_axis = gp_Dir(*rotation_axis)

        axis = gp_Ax1(rotation_point, rotation_axis)

        trsf = gp_Trsf()
        trsf.SetRotation(axis, angle)
        return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


    # ===========================================================
    # entity sample
    # 2d entity
    @staticmethod
    @multi_input
    def add_rectangle(x_min: float, y_min: float, z_min: float, dx: float, dy: float) -> TopoDS_Face:
        """
        创建一个矩形
        Parameters
        ----------
        x_min: float 矩形左下角x坐标
        y_min: float 矩形左下角y坐标
        z_min: float 矩形左下角z坐标
        dx: float 矩形宽度
        dy: float 矩形高度

        Returns
        -------
        OCC 格式的面
        """
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.GC import GC_MakeSegment
        from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeWire,
                                             BRepBuilderAPI_MakeFace,
                                             BRepBuilderAPI_MakeEdge)
        # 定义矩形的四个顶点
        p1 = gp_Pnt(x_min, y_min, z_min)  # 左下角坐标
        p2 = gp_Pnt(x_min + dx, y_min, z_min)  # 右下角
        p3 = gp_Pnt(x_min + dx, y_min + dy, z_min)  # 右上角
        p4 = gp_Pnt(x_min, y_min + dy, z_min)  # 左上角

        # 创建边线（Edge）并添加到线框（Wire）
        wire_builder = BRepBuilderAPI_MakeWire()
        # 底边：p1 -> p2
        edge1 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(p1, p2).Value()).Edge()
        wire_builder.Add(edge1)
        # 右边：p2 -> p3
        edge2 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(p2, p3).Value()).Edge()
        wire_builder.Add(edge2)
        # 顶边：p3 -> p4
        edge3 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(p3, p4).Value()).Edge()
        wire_builder.Add(edge3)
        # 左边：p4 -> p1
        edge4 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(p4, p1).Value()).Edge()
        wire_builder.Add(edge4)

        # 生成面
        return BRepBuilderAPI_MakeFace(wire_builder.Wire()).Face()

    @staticmethod
    @multi_input
    def add_disk(xc: float, yc: float, zc: float, rx: float, ry: float) -> TopoDS_Face:
        """
        创建一个椭圆
        Parameters
        ----------
        xc: float 椭圆圆心 x 坐标
        yc: float 椭圆圆心 y 坐标
        zc: float 椭圆圆心 z 坐标
        rx: float 椭圆 x 半径
        ry: float 椭圆 y 半径

        Returns
        -------

        """
        from math import pi
        from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
        from OCC.Core.Geom import Geom_Ellipse
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace,
        )
        # 1. 定义椭圆坐标系（XY平面，Z方向为法向）
        ellipse_axis = gp_Ax2(
            gp_Pnt(xc, yc, zc),  # 圆心位置
            gp_Dir(0, 0, 1)  # 法线方向（Z轴）
        )

        # 2. 创建椭圆几何体
        if rx < ry:
            rx, ry = ry, rx
        geom_ellipse = Geom_Ellipse(ellipse_axis, rx, ry)

        # 3. 生成完整椭圆边（角度范围 0~2π）
        edge = BRepBuilderAPI_MakeEdge(geom_ellipse, 0, 2 * pi).Edge()

        # 4. 构建闭合线框
        wire = BRepBuilderAPI_MakeWire(edge).Wire()

        # 5. 生成面
        return BRepBuilderAPI_MakeFace(wire).Face()

    @staticmethod
    @multi_input
    def add_circle(xc: float, yc: float, zc: float, r: float) -> TopoDS_Face:
        """
        创建一个圆
        Parameters
        ----------
        xc: float 圆心 x 坐标
        yc: float 圆心 y 坐标
        zc: float 圆心 z 坐标
        r: float 圆半径

        Returns
        -------

        """
        return OCCAdapter.add_disk(xc, yc, zc, r, r)

    @staticmethod
    @multi_input
    def add_ring(
            xc: float,
            yc: float,
            zc: float,
            r_inner: float,
            r_outer: float
    ) -> TopoDS_Face:
        """
        创建二维圆环面（平面环形区域）

        Parameters
        ----------
        xc : float
            圆心X坐标
        yc : float
            圆心Y坐标
        zc : float
            圆心Z坐标
        r_inner : float
            内半径（必须 > 0 且 < r_outer）
        r_outer : float
            外半径（必须 > 0）

        Returns
        -------
        TopoDS_Face
            环形面（二维）

        Examples
        --------
        >>> ring = add_ring(0, 0, 0, 3, 5)
        """
        # 参数校验
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("半径必须为正数")
        if r_inner >= r_outer:
            raise ValueError("内半径必须小于外半径")
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut

        # 创建外圆和内圆面
        outer_disk = OCCAdapter.add_disk(xc, yc, zc, r_outer, r_outer)
        inner_disk = OCCAdapter.add_disk(xc, yc, zc, r_inner, r_inner)

        # 布尔差集操作（外圆 - 内圆 = 环形面）
        ring_face = BRepAlgoAPI_Cut(outer_disk, inner_disk).Shape()
        if not isinstance(ring_face, TopoDS_Shape):
            raise RuntimeError("环形面创建失败")
        return ring_face

    # 3d entity
    @staticmethod
    @multi_input
    def add_box(
            x_min: float,
            y_min: float,
            z_min: float,
            dx: float,
            dy: float,
            dz: float
    ) -> TopoDS_Solid:
        """
        创建立方体/长方体

        Parameters
        ----------
        x_min : float
            左下角X坐标
        y_min : float
            左下角Y坐标
        z_min : float
            左下角Z坐标
        dx : float
            X轴方向长度
        dy : float
            Y轴方向长度
        dz : float
            Z轴方向长度

        Returns
        -------
        TopoDS_Solid
            生成的立方体实体

        Examples
        --------
        >>> box = add_box(0, 0, 0, 10, 20, 5)
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("尺寸参数必须为正数")

        corner1 = gp_Pnt(x_min, y_min, z_min)
        corner2 = gp_Pnt(x_min + dx, y_min + dy, z_min + dz)
        box = BRepPrimAPI_MakeBox(corner1, corner2).Solid()
        return box

    @staticmethod
    @multi_input
    def add_ellipsoid(
            xc: float,
            yc: float,
            zc: float,
            rx: float,
            ry: float,
            rz: float
    ) -> TopoDS_Shape:
        """
        创建椭球体（通过缩放球体实现）

        Parameters
        ----------
        xc : float
            椭球中心X坐标
        yc : float
            椭球中心Y坐标
        zc : float
            椭球中心Z坐标
        rx : float
            X轴半径（必须 > 0）
        ry : float
            Y轴半径（必须 > 0）
        rz : float
            Z轴半径（必须 > 0）

        Returns
        -------
        TopoDS_Solid
            生成的椭球实体

        Examples
        --------
        >>> ellipsoid = add_ellipsoid(0, 0, 0, 5, 3, 2)
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform
        from OCC.Core.gp import gp_Trsf, gp_GTrsf

        # 参数校验
        if rx <= 0 or ry <= 0 or rz <= 0:
            raise ValueError("半径必须为正数")

        # 1. 创建单位球体
        sphere = BRepPrimAPI_MakeSphere(1.0).Solid()

        # 2. 定义缩放变换矩阵
        gtrsf = gp_GTrsf()
        scaling_matrix = gp_Mat(
            gp_XYZ(rx, 0, 0),   # X轴缩放
            gp_XYZ(0, ry, 0),   # Y轴缩放
            gp_XYZ(0, 0, rz)    # Z轴缩放
        )
        gtrsf.SetVectorialPart(scaling_matrix)  # 关键修正：传入3x3矩阵

        # 3. 设置平移（将缩放后的球体移动到目标中心）
        gtrsf.SetTranslationPart(gp_XYZ(xc, yc, zc))

        # 4. 应用变换
        transformed_shape = BRepBuilderAPI_GTransform(sphere, gtrsf).Shape()

        return transformed_shape

    @staticmethod
    @multi_input
    def add_sphere(xc: float, yc: float, zc: float, radius: float) -> TopoDS_Solid:
        """
        创建球体

        Parameters
        ----------
        xc : float
            球心X坐标
        yc : float
            球心Y坐标
        zc : float
            球心Z坐标
        radius : float
            球体半径（必须 > 0）

        Returns
        -------
        TopoDS_Solid
            生成的球体实体

        Examples
        --------
        >>> sphere = add_sphere(0, 0, 0, 5)
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
        if radius <= 0:
            raise ValueError("半径必须为正数")
        return OCCAdapter.add_ellipsoid(xc, yc, zc, radius, radius, radius)

    @staticmethod
    @multi_input
    def add_cylinder(
            xc: float,
            yc: float,
            zc: float,
            radius: float,
            height: float,
            *,
            axis: Tuple[float, float, float] = (0, 0, 1)
    ) -> TopoDS_Solid:
        """
        创建圆柱体（支持任意轴向）

        Parameters
        ----------
        xc : float
            底面中心X坐标
        yc : float
            底面中心Y坐标
        zc : float
            底面中心Z坐标
        radius : float
            底面半径（必须 > 0）
        height : float
            圆柱高度（必须 > 0）
        axis : Tuple[float, float, float], optional
            圆柱轴向，默认沿Z轴

        Returns
        -------
        TopoDS_Solid
            生成的圆柱实体
        """
        if radius <= 0 or height <= 0:
            raise ValueError("半径和高度必须为正数")
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        # 创建默认Z轴方向的圆柱
        ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*axis))
        cylinder = BRepPrimAPI_MakeCylinder(ax, radius, height).Solid()

        # 调整方向并移动到指定位置
        return OCCAdapter.translate(cylinder, (xc, yc, zc))

    @staticmethod
    @multi_input
    def add_torus(
            xc: float,
            yc: float,
            zc: float,
            major_r: float,
            minor_r: float
    ) -> TopoDS_Solid:
        """
        创建三维圆环体（甜甜圈形状）

        Parameters
        ----------
        xc : float
            中心X坐标
        yc : float
            中心Y坐标
        zc : float
            中心Z坐标
        major_r : float
            主半径（环中心到截面中心的距离，必须 > 0）
        minor_r : float
            截面半径（必须 > 0）

        Returns
        -------
        TopoDS_Solid
            圆环实体

        Examples
        --------
        >>> torus = add_torus(0, 0, 0, 10, 2)
        """
        if major_r <= 0 or minor_r <= 0:
            raise ValueError("半径必须为正数")
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus

        # 定义圆环坐标系（默认在XY平面，法向为Z轴）
        ax = gp_Ax2(gp_Pnt(xc, yc, zc), gp_Dir(0, 0, 1))
        return BRepPrimAPI_MakeTorus(ax, major_r, minor_r).Solid()

    @staticmethod
    @multi_input
    def add_hollow_cylinder(
        xc: float,
        yc: float,
        zc: float,
        outer_radius: float,
        inner_radius: float,
        height: float,
            *,
        axis: Tuple[float, float, float] = (0, 0, 1)
    ) -> TopoDS_Solid:
        """
        创建空心圆柱（外圆柱挖去内圆柱）

        Parameters
        ----------
        xc : float
            圆柱底面中心X坐标
        yc : float
            圆柱底面中心Y坐标
        zc : float
            圆柱底面中心Z坐标
        outer_radius : float
            外圆柱半径（必须 > 0）
        inner_radius : float
            内圆柱半径（必须 > 0 且 < outer_radius）
        height : float
            圆柱高度（必须 > 0）
        axis : Tuple[float, float, float], optional
            圆柱轴向，默认沿Z轴

        Returns
        -------
        TopoDS_Solid
            空心圆柱实体

        Examples
        --------
        >>> hollow_cyl = add_hollow_cylinder(0,0,0, 5, 3, 10)
        """
        # 参数校验
        if outer_radius <= 0 or inner_radius <= 0 or height <= 0:
            raise ValueError("半径和高度必须为正数")
        if inner_radius >= outer_radius:
            raise ValueError("内半径必须小于外半径")
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        # 1. 创建外圆柱
        outer_cyl = OCCAdapter.add_cylinder(
            xc, yc, zc, outer_radius, height, axis=axis
        )

        # 2. 创建内圆柱（与外部同轴）
        inner_cyl = OCCAdapter.add_cylinder(
            xc, yc, zc, inner_radius, height, axis=axis
        )

        # 3. 布尔差集操作（外 - 内）
        result = BRepAlgoAPI_Cut(outer_cyl, inner_cyl).Shape()
        if not isinstance(result, TopoDS_Shape):
            raise RuntimeError("空心圆柱创建失败")

        # 4. 验证几何有效性
        analyzer = BRepCheck_Analyzer(result)
        if not analyzer.IsValid():
            raise RuntimeError("生成的几何体无效")

        return result


    # ===========================================================
    # entity sample
    @staticmethod
    def shape_discrete(shape: TopoDS_Shape, deflection: float=0.1):
        """
        对 OCC 的 TopoDS_Shape 进行网格化，并提取顶点和三角形面片数据

        Parameters
        ----------
        shape : TopoDS_Shape
            待网格化的几何体
        deflection : float, optional
            网格离散化的精度参数，数值越小网格越细，默认 0.1

        Returns
        -------
        vertices : list of tuple(float, float, float)
            所有顶点的坐标列表
        triangles : list of tuple(int, int, int)
            三角形面片，使用顶点索引表示（注意：OCC 中顶点索引从 1 开始，需要转换为 0 索引）
        """
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRep import BRep_Tool
        # 对形状进行网格化
        mesh = BRepMesh_IncrementalMesh(shape, deflection)
        mesh.Perform()

        vertices = []
        triangles = []
        vertex_offset = 0  # 用于记录全局顶点索引偏移

        # 遍历形状中的所有面
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            # 获取该面的三角化数据
            triangulation = BRep_Tool.Triangulation(face, face.Location())
            if triangulation is not None:
                # 获取顶点数
                num_nodes = triangulation.NbNodes()
                local_vertices = []
                for i in range(1, num_nodes + 1):  # OCC 的索引从 1 开始
                    pnt = triangulation.Node(i)  # 使用 Node(i) 逐个获取点
                    local_vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
                # 将当前面的顶点添加到全局列表
                vertices.extend(local_vertices)

                # 获取三角形面数
                num_triangles = triangulation.NbTriangles()
                for i in range(1, num_triangles + 1):
                    triangle = triangulation.Triangle(i)  # 获取第 i 个三角形
                    # 获取面片顶点的索引
                    n1, n2, n3 = triangle.Get()
                    # 转换为 0 索引，并加上当前顶点偏移量
                    triangles.append((n1 - 1 + vertex_offset,
                                      n2 - 1 + vertex_offset,
                                      n3 - 1 + vertex_offset))
                # 更新全局顶点偏移
                vertex_offset = len(vertices)
            explorer.Next()

        return vertices, triangles



    # ===========================================================
    # file io
    @staticmethod
    def import_step(filename: Union[str, Path]) -> TopoDS_Shape:
        """Import a STEP file and return a single shape.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the STEP file (.stp or .step).

        Returns
        -------
        TopoDS_Shape
            The shape extracted from the STEP file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If the STEP file is invalid or cannot be read.
        """
        from OCC.Core.STEPControl import STEPControl_Reader

        filename = str(filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"STEP file not found: {filename}")

        reader = STEPControl_Reader()
        status = reader.ReadFile(filename)
        if status != 1:  # Check if reading was successful
            raise RuntimeError(f"Failed to read STEP file: {filename}")

        reader.TransferRoots()
        shape = reader.OneShape()  # Get a single composite shape
        if shape.IsNull():
            raise RuntimeError(f"No valid shape found in STEP file: {filename}")

        return shape

    @staticmethod
    def import_stl(filename: Union[str, Path]) -> TopoDS_Shape:
        """Import an STL file and return a single shape.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the STL file (.stl, supports ASCII or binary format).

        Returns
        -------
        TopoDS_Shape
            The shape representing the STL mesh.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If the STL file is invalid or cannot be read.
        """
        from OCC.Core.StlAPI import StlAPI_Reader

        filename = str(filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"STL file not found: {filename}")

        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        success = reader.Read(shape, filename)
        if not success or shape.IsNull():
            raise RuntimeError(f"Failed to read STL file: {filename}")

        return shape

    @staticmethod
    def import_brep(filename: Union[str, Path]) -> 'TopoDS_Shape':
        """Import a BREP file and return a single shape.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the BREP file (.brep).

        Returns
        -------
        TopoDS_Shape
            The shape representing the BREP geometry.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        RuntimeError
            If the BREP file is invalid, empty, or cannot be read.
        """
        # 局部导入
        from OCC.Core.BRepTools import breptools_Read # type: ignore
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.BRep import BRep_Builder

        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"BREP file not found: {filename}")

        builder = BRep_Builder()
        shape = TopoDS_Shape()
        success = breptools_Read(shape, str(path), builder)

        if not success or shape.IsNull():
            raise RuntimeError(f"Failed to read BREP file or the shape is null: {filename}")

        return shape

    @staticmethod
    def export_step(*shape: TopoDS_Shape, filename: Union[str, Path]) -> None:
        """Export multiple shapes to a STEP file.

        Parameters
        ----------
        *shape : TopoDS_Shape
            Variable number of shapes to export, merged into a composite shape.
        filename : Union[str, Path]
            Path to the output STEP file (.stp or .step).

        Raises
        ------
        ValueError
            If no shapes are provided or any shape is invalid.
        FileNotFoundError
            If the output directory is inaccessible.
        RuntimeError
            If the STEP export process fails.
        """
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.BRep import BRep_Builder

        if not shape:
            raise ValueError("No shapes provided for STEP export")

        filename = str(filename)
        output_dir = os.path.dirname(filename) or "."
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Output directory inaccessible: {output_dir}")

        # Merge shapes into a compound
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for s in shape:
            if s.IsNull():
                raise ValueError("Invalid shape provided for STEP export")
            builder.Add(compound, s)

        # Export to STEP
        writer = STEPControl_Writer()
        writer.Transfer(compound, STEPControl_AsIs)
        status = writer.Write(filename)
        if status != 1:  # Check if writing was successful
            raise RuntimeError(f"Failed to export STEP file: {filename}")

    @staticmethod
    def export_stl(
            *shape: TopoDS_Shape, filename: Union[str, Path], resolution: float = 0.1
    ) -> None:
        """Export multiple shapes to an STL file with mesh resolution control.

        Parameters
        ----------
        *shape : TopoDS_Shape
            Variable number of shapes to export, merged into a composite shape.
        filename : Union[str, Path]
            Path to the output STL file (.stl, supports ASCII or binary format).
        resolution : float, optional
            Linear deflection for meshing (smaller is finer, default is 0.1).

        Raises
        ------
        ValueError
            If no shapes are provided, any shape is invalid, or resolution <= 0.
        FileNotFoundError
            If the output directory is inaccessible.
        RuntimeError
            If the STL export process fails.
        """
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

        if not shape:
            raise ValueError("No shapes provided for STL export")
        if resolution <= 0:
            raise ValueError("Resolution must be positive")

        filename = str(filename)
        output_dir = os.path.dirname(filename) or "."
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Output directory inaccessible: {output_dir}")

        # Merge shapes into a compound
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for s in shape:
            if s.IsNull():
                raise ValueError("Invalid shape provided for STL export")
            builder.Add(compound, s)

        # Mesh the compound
        mesh = BRepMesh_IncrementalMesh(compound, resolution)
        mesh.Perform()
        if not mesh.IsDone():
            raise RuntimeError("Meshing failed for STL export")

        # Export to STL
        writer = StlAPI_Writer()
        writer.SetASCIIMode(True)  # Use ASCII format for compatibility
        success = writer.Write(compound, filename)
        if not success:
            raise RuntimeError(f"Failed to export STL file: {filename}")

    @staticmethod
    def export_brep(*shape: TopoDS_Shape, filename: Union[str, Path]) -> None:
        """Export multiple shapes to a BREP file.

        Parameters
        ----------
        *shape : TopoDS_Shape
            Variable number of shapes to export, merged into a composite shape.
        filename : Union[str, Path]
            Path to the output BREP file (.brep).

        Raises
        ------
        ValueError
            If no shapes are provided or any shape is invalid.
        FileNotFoundError
            If the output directory is inaccessible.
        RuntimeError
            If the BREP export process fails.
        """
        from OCC.Core.TopTools import TopTools_ListOfShape
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepTools import breptools_Write # type: ignore

        # 校验输入
        if not shape:
            raise ValueError("At least one shape must be provided for export.")
        for s in shape:
            if s.IsNull():
                raise ValueError("One of the provided shapes is null/invalid.")

        filename = Path(filename)
        if not filename.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {filename.parent}")

        # 构建复合体（Compound）
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s in shape:
            builder.Add(compound, s)

        # 写入 BREP 文件
        success = breptools_Write(compound, str(filename))
        if not success:
            raise RuntimeError(f"Failed to write BREP file: {filename}")

    # ===========================================================
    # display
    @staticmethod
    def display(
            *shapes: Any,
            background: Union[str, Sequence[float]] = "white",  # 支持颜色名称、简写、十六进制
            window_size: tuple = (1024, 768),  # 窗口大小 (width, height)
            show_axes: bool = True,  # 是否显示坐标系
            zoom_all: bool = True,  # 自动缩放适应所有模型
            **kwargs  # 其他显示参数（如透明度、颜色）
    ) -> None:
        """
        可视化一个或多个 OCC 模型

        Parameters
        ----------
        *shapes : OCC.TopoDS_Shape
            要显示的几何体（支持多个）
        background : str, optional
            窗口背景颜色，默认为白色
        window_size : tuple, optional
            窗口分辨率，默认 (1024, 768)
        show_axes : bool, optional
            是否显示坐标系，默认 True
        zoom_all : bool, optional
            是否自动缩放，默认 True
        **kwargs : dict
            高级参数：
            - color: Union[str, Sequence[str]]
                颜色名称或列表（如 "red", ["#FF0000", "#00FF00"]）
            - transparency: Union[float, Sequence[float]]
                透明度（0.0~1.0）
        """
        try:
            from OCC.Display.SimpleGui import init_display
            from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
            import matplotlib.colors as mcolors  # 依赖 matplotlib
        except ImportError as e:
            raise RuntimeError("依赖未安装：请安装 pythonocc-core 和 matplotlib。") from e

        if not shapes:
            raise ValueError("至少需要传入一个几何体进行显示。")

        # 转换背景颜色为 OCC 所需的 RGB 整数格式
        def to_occ_rgb(color: Union[str, Sequence[float]]) -> list:
            rgb = parse_color(color)
            return [int(c * 255) for c in rgb]

            # 初始化显示窗口（参数名称已修正！）

        display, start_display, add_menu, _ = init_display(
            size=window_size,
            background_gradient_color1=to_occ_rgb(background),
            background_gradient_color2=to_occ_rgb(background),
            display_triedron=show_axes
        )

        # 处理颜色和透明度参数（同上）
        colors = kwargs.get("color", ["steelblue"])  # 默认颜色
        transparencies = kwargs.get("transparency", [0.0])

        # 统一为列表格式
        if isinstance(colors, (str, tuple, list)):
            colors = [colors] * len(shapes) if isinstance(colors, (str, tuple)) else colors
        if isinstance(transparencies, (float, int)):
            transparencies = [transparencies] * len(shapes)

        for idx, shape in enumerate(shapes):
            try:
                # 解析颜色
                rgb = parse_color(colors[idx % len(colors)])
                color = Quantity_Color()
                color.SetValues(*rgb, Quantity_TOC_RGB)
            except ValueError as e:
                raise ValueError(f"颜色参数错误：{e}")
            transparency = transparencies[idx % len(transparencies)]

            display.DisplayShape(
                shape,
                color=color,
                transparency=transparency,
                update=(idx == len(shapes) - 1)
            )

        if zoom_all:
            display.FitAll()

        start_display()

    # ===========================================================



attribute_mapping = ATTRIBUTE_MAPPING.copy()
function_mapping = FUNCTION_MAPPING.copy()

OCCAdapter.attach_attributes(attribute_mapping, OCC)
OCCAdapter.attach_methods(function_mapping, OCC)